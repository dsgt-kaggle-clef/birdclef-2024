import pandas as pd
import math

# Sample JSON data
json_data = [
    {
        "id": 33883,
        "annotations": [
            {
                "id": 107,
                "result": [
                    {
                        "value": {
                            "start": 1.7884691987767585,
                            "end": 3.06814974617737,
                            "labels": ["Primary Call"]
                        }
                    }
                ]
            }
        ],
        "data": {
            "audio": "gs:\/\/dsgt-clef-birdclef-2024\/data\/processed\/birdclef-2023\/truncated_audio\/abethr1\/XC128013.mp3"
        }
    },
    {
        "id": 33884,
        "annotations": [
            {
                "id": 108,
                "result": [
                    {
                        "value": {
                            "start": 1.7451806538461538,
                            "end": 2.463361992877493,
                            "labels": ["Primary Call"]
                        }
                    },
                    {
                        "value": {
                            "start": 3.124088824786325,
                            "end": 5.041633,
                            "labels": ["Secondary Call"]
                        }
                    }
                ]
            }
        ],
        "data": {
            "audio": "gs:\/\/dsgt-clef-birdclef-2024\/data\/processed\/birdclef-2023\/truncated_audio\/abethr1\/XC363501.mp3"
        }
    },
    {
        "id": 33885,
        "annotations": [
            {
                "id": 109,
                "result": [
                    {
                        "value": {
                            "start": 3.2892705327635325,
                            "end": 4.775905904558405,
                            "labels": ["Primary Call"]
                        }
                    },
                    {
                        "value": {
                            "start": 1.4076354245014244,
                            "end": 3.037907064102564,
                            "labels": ["Secondary Call"]
                        }
                    }
                ]
            }
        ],
        "data": {
            "audio": "gs:\/\/dsgt-clef-birdclef-2024\/data\/processed\/birdclef-2023\/truncated_audio\/abethr1\/XC363502.mp3"
        }
    }
]

# Function to process JSON data and create DataFrame
def process_json_to_df(json_data):
    df_list = []
    for item in json_data:
        audio_file = item['data']['audio']
        annotations = item['annotations']
        for annotation in annotations:
            results = annotation['result']
            for result in results:
                start = math.floor(result['value']['start'])
                end = math.ceil(result['value']['end'])
                labels = result['value']['labels']
                for i in range(start, end):
                    primary_call = 'Primary Call' in labels
                    secondary_call = 'Secondary Call' in labels
                    df_list.append({'audio': audio_file, 'interval': i, 'primary_call': primary_call, 'secondary_call': secondary_call})
    df = pd.DataFrame(df_list)
    return df

df = process_json_to_df(json_data)
print(df)
df.to_csv('annotations.csv', index=False)