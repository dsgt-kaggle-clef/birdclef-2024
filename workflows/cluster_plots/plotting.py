from pathlib import Path

import luigi
import matplotlib.pyplot as plt

from birdclef.knn_labels import (
    compute_embedding_2d,
    get_knn_labels,
    get_label_agreement,
    get_subset_pdf,
    plot_distances,
    plot_embedding,
    write_plots_to_disk,
)
from birdclef.utils import get_spark


class ClusterPlotAllTasks(luigi.WrapperTask):
    spark_path = luigi.Parameter()
    local_path = luigi.Parameter()
    total_cnt = luigi.IntParameter()

    def requires(self):
        spark = get_spark(memory="2g")
        df = spark.read.parquet(self.spark_path)
        labeled_neighborhood = get_knn_labels(df).cache()

        for i in range(self.total_cnt):
            pdf = get_subset_pdf(df, labeled_neighborhood, i)
            name = pdf.ego_primary_label.iloc[0]
            path_name = f"{self.local_path}/{name}"
            path = Path(path_name)
            path.mkdir(parents=True, exist_ok=True)

            yield ClusterPlotSingleSpecies(
                local_path=path_name,
                pdf=pdf,
                name=name,
            )


class ClusterPlotSingleSpecies(luigi.WrapperTask):
    local_path = luigi.Parameter()
    pdf = luigi.Parameter()
    name = luigi.Parameter()

    def requires(self):
        distance = DistancePlotTask(
            path=Path(self.local_path),
            pdf=self.pdf,
        )
        yield distance

        ego_birdnet_label = EgoBirdnetLabelTask(
            path=Path(self.local_path),
            pdf=self.pdf,
        )
        yield ego_birdnet_label

        knn_birdnet_label = KnnBirdnetLabelTask(
            path=Path(self.local_path),
            pdf=self.pdf,
        )
        yield knn_birdnet_label


class DistancePlotTask(luigi.Task):
    path = luigi.Parameter()
    pdf = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.path / "distances.png")

    def run(self):
        plot_distances(self.pdf)
        plt.savefig(self.output().path)
        plt.close()


class EgoBirdnetLabelTask(luigi.Task):
    path = luigi.Parameter()
    pdf = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.path / "ego_birdnet_label.png")

    def run(self):
        emb = compute_embedding_2d(self.pdf)
        plot_embedding(self.pdf, emb, "ego_birdnet_label", 5)
        plt.savefig(self.output().path)
        plt.close()


class KnnBirdnetLabelTask(luigi.Task):
    path = luigi.Parameter()
    pdf = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.path / "knn_birdnet_label.png")

    def run(self):
        emb = compute_embedding_2d(self.pdf)
        plot_embedding(self.pdf, emb, "knn_birdnet_label", 5)
        plt.savefig(self.output().path)
        plt.close()
