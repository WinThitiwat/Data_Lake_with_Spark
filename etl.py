import configparser
import os
import logging

from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek
from pyspark.sql.types import StructType, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Date, LongType as Long, TimestampType
from time import time



def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    song_data = os.path.join(input_data, "song-data/A/A/A/*.json")
    
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
    
    logging.info("Start reading `song_data` json file(s)...")
    start_time = time()
    
    df = spark.read.json(song_data, schema=song_schema).drop_duplicates()
    
    logging.info("Finished reading 'song_data' json file(s)...")
    
    logging.info("Reading 'song_data' json took: {0:.2f} second".format(
        time() - start_time
    ))
    
    # =================================
    # songs_table
    # =================================
    
    # extract columns to create songs table
    logging.info("Start extracting `songs` table...")
    start_time = time()
    
    song_columns = ["song_id", "title", "artist_id", "year", "duration"]
    songs_table = df.select(song_columns).drop_duplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write \
        .mode("overwrite") \
        .partitionBy('year', 'artist_id') \
        .parquet(output_data + "songs/")
    
    logging.info("Finished extracting `songs` table")
    
    logging.info("extracting `songs` table took: {0:.2f} second".format(
        time() - start_time
    ))
    
    # =================================
    # artists_table
    # =================================
    
    # extract columns to create artists table
    artist_columns = ["artist_id", "artist_name as name", "artist_location as location", "artist_longitude as longitude", "artist_latitude as latitude"]
    
    artists_table = df.selectExpr(artist_columns)
    
    # write artists table to parquet files
    artists_table.write \
        .mode("overwrite") \
        .parquet(output_data + "artists/")


def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = os.path.join(input_data, "log-data/2018/11/*.json")
    
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
    logging.info("Start reading 'log_data' json file(s)...")
    
    start_time = time()
    
    df = spark.read.json(log_data, schema=log_schema).drop_duplicates()
    
    logging.info("Finished reading 'log_data' json file(s)...")
    
    logging.info("Reading 'log_data' json took: {0:.2f} second".format(
        time() - start_time
    ))
    
    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")
    
    # =================================
    # users_table
    # =================================
    
    # extract columns for users table 
    user_columns = ["userId", "firstName", "lastName", "gender", "level"]
    users_table = df.select(user_columns)
    
    # write users table to parquet files
    users_table.write \
        .mode("overwrite") \
        .parquet(output_data + "users/")

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
    time_column = ["ts", "hour", "day", "week", "month", "year", "weekday"]
    time_table = df.withColumn("hour", hour("ts_timestamp")) \
                    .withColumn("day", dayofmonth("ts_timestamp")) \
                    .withColumn("week", weekofyear("ts_timestamp")) \
                    .withColumn("month", month("ts_timestamp")) \
                    .withColumn("year", year("ts_timestamp")) \
                    .withColumn("weekday", dayofweek("ts_timestamp")) \
                    .select(time_column).drop_duplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write \
        .mode('overwrite') \
        .partitionBy('year', 'month') \
        .parquet(output_data + "time/")
    
    # =================================
    # songplays_table
    # =================================
    
    # read in song data to use for songplays table
    song_df = spark.read \
                .parquet(output_data + "songs/*/*/*.parquet")

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df.join(song_df,
                             [song_df.title == df.song,
                             song_df.artist_name == df.artist],
                             how='inner')

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write \
        .mode('overwrite') \
        .partitionBy('year', 'month') \
        .parquet(output_data + "songplays/")


def main():
    config = configparser.ConfigParser()
    config.read('dl.cfg')

    os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
    os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']

    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "data/data_outputs/"
    
    process_song_data(spark, input_data, output_data)    
#     process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
