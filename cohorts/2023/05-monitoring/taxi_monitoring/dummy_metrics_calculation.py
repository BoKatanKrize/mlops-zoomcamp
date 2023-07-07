import datetime
import time
import random
import logging 
import uuid          # provide immutable IDs
import pytz          # timezone calculations
import pandas as pd
import io
import psycopg       # accessing the PostgreSQL database

# Sets logging level to INFO (i.e. log messages with INFO
# level and above will be displayed)
logging.basicConfig(level=logging.INFO,
					format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

# SQL statement for creating a table with 4 columns
create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	value1 integer,
	value2 varchar,
	value3 float
)
"""


def prep_db():
	# Establishing a connection to the PostgreSQL server
	# (port=5432 correspond to db service in compose.yaml)
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		# SQL query to check if a database named 'test' already exists
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		# If the result of the query == 0, the database 'test' does not exist
		if len(res.fetchall()) == 0:
			# Then SQL statement creates a new 'test' database
			conn.execute("create database test;")
		# This time, the connection parameters include the database name as well
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			# SQL statement creates a table named 'dummy_metrics'
			conn.execute(create_table_statement)


def calculate_dummy_metrics_postgresql(curr):
	value1 = rand.randint(0, 1000)
	value2 = str(uuid.uuid4())  # Generate a random UUID (Universally Unique Identifier)
	value3 = rand.random()
	# Execute an SQL insert statement to insert the calculated values into the dummy_metrics table
	curr.execute(
		"insert into dummy_metrics(timestamp, value1, value2, value3) values (%s, %s, %s, %s)",
		(datetime.datetime.now(pytz.timezone('Europe/London')), value1, value2, value3)
	)


def main():
	# Prepare the database for usage
	prep_db()
	# Set to the current time minus 10 seconds
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	# Establish a connection to the PostgreSQL database
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		for i in range(0, 100):
			# Creates a cursor object. It allows you to execute SQL queries and interact with the database
			with conn.cursor() as curr:
				# Insert new data in each iteration
				calculate_dummy_metrics_postgresql(curr)

			new_send = datetime.datetime.now()
			# Calculate the number of seconds elapsed since the last send
			seconds_elapsed = (new_send - last_send).total_seconds()
			# If the seconds elapsed is less than the timeout, sleep for the remaining time
			# to ensure the specified timeout is reached before sending the next batch of data
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				# Update last_send by adding 10 seconds to it until it reaches or surpasses new_send
				# This ensures that last_send is as a max 10 seconds ahead of new_send
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")


if __name__ == '__main__':
	main()