import configparser
import os
import logging

from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek
from pyspark.sql.types import StructType, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Date, LongType as Long, TimestampType
from time import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()

    spark.conf.set("mapreduce.fileoutputcommitter.algorithm.version", "2")

    return spark


def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    song_data = os.path.join(input_data, "song-data/*/*/*/*.json")
    
    # =================================
    # read song data file
    # =================================
    
    # This schema is based on conducting data profiling
    song_schema = StructType([
        Fld("artist_id", Str()),
        Fld("artist_latitude", Dbl()),
        Fld("artist_location", Str()),
        Fld("artist_longitude", Dbl()),
        Fld("artist_name", Str()),
        Fld("duration", Dbl()),
        Fld("num_songs", Long()),
        Fld("song_id", Str()),
        Fld("title", Str()),
        Fld("year", Long()),
    ])
    
    logger.info("Start reading `song_data` json file(s)...")
    start_time = time()
    
    df = spark.read.json(song_data, schema=song_schema).drop_duplicates()
    
    logger.info("Finished reading 'song_data' json file(s)...")
    
    logger.info("Reading 'song_data' json took: {0:.2f} seconds".format(
        time() - start_time
    ))
    
    # =================================
    # songs_table
    # =================================
    
    # extract columns to create songs table
    logger.info("Preparing `songs` dataframe")
    
    song_columns = ["song_id", "title", "artist_id", "year", "duration"]
    songs_table = df.select(song_columns).drop_duplicates()
    
    logger.info("Start exporting `songs` parquet files...")
    
    start_time = time()
    # write songs table to parquet files partitioned by year and artist
    songs_table.write \
        .mode("overwrite") \
        .partitionBy('year', 'artist_id') \
        .parquet(output_data + "songs/")
    
    logger.info("Finished exporting `songs` parquet files")
    
    logger.info("Exporting `songs` parquet files took: {0:.2f} seconds".format(
        time() - start_time
    ))
    
    # =================================
    # artists_table
    # =================================
    
    # extract columns to create artists table
    logger.info("Preparing `artists` dataframe")

    artist_columns = ["artist_id", "artist_name as name", "artist_location as location", "artist_longitude as longitude", "artist_latitude as latitude"]
    
    artists_table = df.selectExpr(artist_columns)
    
    logger.info("Start exporting `artists` parquet files...")
    
    start_time = time()
    # write artists table to parquet files
    artists_table.write \
        .mode("overwrite") \
        .parquet(output_data + "artists/")
    
    logger.info("Finished exporting `artists` parquet files")
    
    logger.info("Exporting `artists` parquet files took: {0:.2f} seconds".format(
        time() - start_time
    ))

def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = os.path.join(input_data, "log-data/*/*/*.json")
    
    # =================================
    # read log data file
    # =================================

    log_schema = StructType([
        Fld("artist", Str()),
        Fld("auth", Str()),
        Fld("firstName", Str()),
        Fld("gender", Str()),
        Fld("itemInSession", Long()),
        Fld("lastName", Str()),
        Fld("length", Dbl()),
        Fld("level", Str()),
        Fld("location", Str()),
        Fld("method", Str()),
        Fld("page", Str()),
        Fld("registration", Dbl()),
        Fld("sessionId", Long()),
        Fld("song", Str()),
        Fld("status", Int()),
        Fld("ts", Long()),
        Fld("userAgent", Str()),
        Fld("userId", Str())
    ])
    logger.info("Start reading `log_data` json file(s)...")
    
    start_time = time()
    
    df = spark.read.json(log_data, schema=log_schema).drop_duplicates()
    
    logger.info("Finished reading 'log_data' json file(s)...")
    
    logger.info("Reading 'log_data' json took: {0:.2f} seconds".format(
        time() - start_time
    ))
    
    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")
    
    # =================================
    # users_table
    # =================================
    
    # extract columns for users table 
    logger.info("Preparing `users` dataframe")

    user_columns = ["userId", "firstName", "lastName", "gender", "level"]
    
    users_table = df.select(user_columns)
    
    logger.info("Start exporting `users` parquet files...")
    
    start_time = time()

    # write users table to parquet files
    users_table.write \
        .mode("overwrite") \
        .parquet(output_data + "users/")
    
    logger.info("Finished exporting `users` parquet files")
    
    logger.info("Exporting `users` parquet files took: {0:.2f} seconds".format(
        time() - start_time
    ))

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.utcfromtimestamp(x/1000.0), TimestampType())
    df = df.withColumn("ts_timestamp", get_timestamp("ts"))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.utcfromtimestamp(x/1000.0).strftime('%Y-%m-%d %H:%M:%S'))
    df = df.withColumn("ts_datetime", get_datetime("ts"))

    # =================================
    # time_table
    # =================================
    
    # extract columns to create time table
    logger.info("Preparing `time` dataframe")

    time_column = ["ts", "ts_datetime as start_time", "hour", "day", "week", "month", "year", "weekday"]
    time_table = df.withColumn("hour", hour("ts_timestamp")) \
                    .withColumn("day", dayofmonth("ts_timestamp")) \
                    .withColumn("week", weekofyear("ts_timestamp")) \
                    .withColumn("month", month("ts_timestamp")) \
                    .withColumn("year", year("ts_timestamp")) \
                    .withColumn("weekday", dayofweek("ts_timestamp")) \
                    .selectExpr(time_column).drop_duplicates()
    
    logger.info("Start exporting `time` parquet files...")
    
    start_time = time()
    # write time table to parquet files partitioned by year and month
    time_table.write \
        .mode('overwrite') \
        .partitionBy('year', 'month') \
        .parquet(output_data + "time/")
    
    logger.info("Finished exporting `time` parquet files")
    
    logger.info("Exporting `time` parquet files took: {0:.2f} seconds".format(
        time() - start_time
    ))
    # =================================
    # songplays_table
    # =================================
    
    # read in song data to use for songplays table
    song_input = os.path.join(input_data, "song-data/*/*/*/*.json")

    logger.info("Start reading `song_df` json file(s)...")

    song_df = spark.read.json(song_input)

    # extract columns from joined song and log datasets to create songplays table 
    # join with song_df
    logger.info("Start joining `song_df` and log datasets...")

    songplays_table = df.join(song_df,
                             [song_df.title == df.song,
                             song_df.artist_name == df.artist],
                             how='inner') \
                        .select([monotonically_increasing_id().alias("songplay_id"), 
                            col("ts_datetime").alias("start_time"), 
                            "userId", 
                            "level", 
                            "song_id", 
                            "artist_id", 
                            "sessionId", 
                            "location", 
                            "userAgent"])

    # join with time_table to extract month and year
    songplays_table = songplays_table.join(time_table,
                                        [songplays_table.start_time == time_table.start_time],
                                        how='inner')\
                                    .select(
                                        "songplay_id",
                                        songplays_table.start_time, 
                                        "userId", 
                                        "level", 
                                        "song_id", 
                                        "artist_id", 
                                        "sessionId", 
                                        "location", 
                                        "userAgent",
                                        "month",
                                        "year"
                                    )


    logger.info("Start exporting `songplays` parquet files...")
    
    start_time = time()

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write \
        .mode('overwrite') \
        .partitionBy('year', 'month') \
        .parquet(output_data + "songplays/")
    
    logger.info("Finished exporting `songplays` parquet files")
    
    logger.info("Exporting `songplays` parquet files took: {0:.2f} seconds".format(
        time() - start_time
    ))

def main():

    start_process = time()
    spark = create_spark_session()
    input_data = config['S3']['INPUT_DATA']
    output_data = config['S3']['OUTPUT_DATA']
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)

    spark.stop()

    logger.info("Loading All Data completed!")

    logger.info("Total time took: {0:.2f} seconds".format(
        time() - start_process
    ))


if __name__ == "__main__":
    main()
