import io
import os

import numpy as np
import torch
from PIL import Image
from pyspark.ml import Transformer
from pyspark.ml.functions import predict_batch_udf
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType


class SpeciesData:
    def get_species(self) -> list:
        species_text = """
        asbfly, ashdro1, ashpri1, ashwoo2, asikoe2, asiope1, aspfly1, aspswi1, barfly1, barswa,
        bcnher, bkcbul1, bkrfla1, bkskit1, bkwsti, bladro1, blaeag1, blakit1, blhori1, blnmon1,
        blrwar1, bncwoo3, brakit1, brasta1, brcful1, brfowl1, brnhao1, brnshr, brodro1, brwjac1,
        brwowl1, btbeat1, bwfshr1, categr, chbeat1, cohcuc1, comfla1, comgre, comior1, comkin1,
        commoo3, commyn, compea, comros, comsan, comtai1, copbar1, crbsun2, cregos1, crfbar1,
        crseag1, dafbab1, darter2, eaywag1, emedov2, eucdov, eurbla2, eurcoo, forwag1, gargan,
        gloibi, goflea1, graher1, grbeat1, grecou1, greegr, grefla1, grehor1, grejun2, grenig1,
        grewar3, grnsan, grnwar1, grtdro1, gryfra, grynig2, grywag, gybpri1, gyhcaf1, heswoo1,
        hoopoe, houcro1, houspa, inbrob1, indpit1, indrob1, indrol2, indtit1, ingori1, inpher1,
        insbab1, insowl1, integr, isbduc1, jerbus2, junbab2, junmyn1, junowl1, kenplo1, kerlau2,
        labcro1, laudov1, lblwar1, lesyel1, lewduc1, lirplo, litegr, litgre1, litspi1, litswi1,
        lobsun2, maghor2, malpar1, maltro1, malwoo1, marsan, mawthr1, moipig1, nilfly2, niwpig1,
        nutman, orihob2, oripip1, pabflo1, paisto1, piebus1, piekin1, placuc3, plaflo1, plapri1,
        plhpar1, pomgrp2, purher1, pursun3, pursun4, purswa3, putbab1, redspu1, rerswa1, revbul,
        rewbul, rewlap1, rocpig, rorpar, rossta2, rufbab3, ruftre2, rufwoo2, rutfly6, sbeowl1,
        scamin3, shikra1, smamin1, sohmyn1, spepic1, spodov, spoowl1, sqtbul1, stbkin1, sttwoo1,
        thbwar1, tibfly3, tilwar1, vefnut1, vehpar1, wbbfly1, wemhar1, whbbul2, whbsho3, whbtre1,
        whbwag1, whbwat1, whbwoo2, whcbar1, whiter2, whrmun, whtkin2, woosan, wynlau1, yebbab1,
        yebbul3, zitcis1
        """
        species = [
            s.strip() for s in species_text.strip().replace("\n", "").split(", ")
        ]
        return species

    def write_species_to_file(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(root_dir, "birdclef", "species.txt")
        species = self.get_species()
        with open(file_path, "w") as f:
            for specie in species:
                f.write(f"{specie}\n")

    def read_species_from_file(self) -> list:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(root_dir, "birdclef", "species.txt")
        with open(file_path, "r") as f:
            species = f.read().splitlines()
        return species


class TransformEmbedding(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Wrapper for transforming the embedding DataFrame and getting its species
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        batch_size: int = 8,
    ):
        super().__init__()
        self._setDefault(inputCol=input_col, outputCol=output_col)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = SpeciesData()
        self.species = sd.read_species_from_file()

    def sigmoid_udf(self, x):
        """
        compute the sigmoid of the columns and only keep
        the rows that are above a certain threshold
        """
        return 1 / (1 + F.exp(-F.col(x)))

    def _transform(self, df, k: int = 1.4826):
        exploded = (
            df.select(
                F.concat_ws("_", "name", "chunk_5s").alias("id"),
                F.arrays_zip("logits", F.lit(self.species).alias("species")).alias(
                    "logits"
                ),
            )
            .withColumn("exploded", F.explode("logits"))
            .select("id", "exploded.*")
            .withColumn("sigmoid", self.sigmoid_udf("logits"))
        ).cache()
        # median absolute deviation of the sigmoid column
        median = exploded.approxQuantile("sigmoid", [0.5], 0.001)[0]
        mad = (
            exploded.withColumn(
                "diff", F.abs(F.col("sigmoid") - median)
            ).approxQuantile("diff", [0.5], 0.001)[0]
            * k
        )
        print(median, mad)
        scored = exploded.withColumn("score", (F.col("sigmoid") - median) / mad).cache()
        scored.withColumn("keep", F.col("score") > 3).groupBy("keep").count().show()
        # predictions with sigmoid > 0.5
        predictions = (
            scored.where("sigmoid > 0.5")
            .groupBy("id")
            .agg(F.collect_list("species").alias("predicted"))
        )
        # rows without predictions out of the total
        scored.select("id").distinct().join(
            predictions.select("id"), on="id", how="anti"
        ).count(), scored.select("id").distinct().count()

        return scored
