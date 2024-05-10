from diagrams.aws.storage import S3
from diagrams.aws.analytics import EMR
from diagrams.onprem.analytics import Spark
from diagrams import Cluster, Diagram
from diagrams.c4 import Database

graph_attr = {
    "splines": "spline",
}

with Diagram("System Architecture", show=False, graph_attr=graph_attr):
    

    with Cluster("S3 Raw Event Store"):
        stg_data_lake = [
            S3("song-data-json"),
            S3("log-data-json")
        ]

    with Cluster("Data Processing"):
        pyspark_file = Spark("PySpark Job Summitter") 
        data_processing = EMR("EMR for Spark")
    
    with Cluster("S3 Data Lake"):

        with Cluster("Staging"):
            stg_song_data = Database("stg_songs_data")
            stg_log_data = Database("stg_users_activity_log")

        with Cluster("Prod"):
            fct_songplay_db = Database("fct_songplay")

            dim_songs = Database("dim_songs")
            dim_artists = Database("dim_artists")
            dim_time = Database("dim_time")
            dim_users = Database("dim_users")
            
    stg_song_data >> [dim_songs, dim_artists]

    stg_log_data >> [dim_users, dim_time]
    
    [dim_songs, dim_artists, dim_users, dim_time] >> fct_songplay_db


    stg_data_lake >> pyspark_file >> data_processing >> [stg_song_data,stg_log_data]
    