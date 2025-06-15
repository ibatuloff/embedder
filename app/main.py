import ollama
import os
import logging
import psycopg2
from contextlib import contextmanager
from dotenv import load_dotenv
import time
# ~~~~~~~~~~~~~~ logger ~~~~~~~~~~~~~~ #

logging.basicConfig(
    handlers=[
        logging.StreamHandler() 
    ],
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()

# ~~~~~~~~~~~~~~ embedding generation ~~~~~~~~~~~~~~ #
load_dotenv()
conn_data = {
    'host': os.getenv('HOST1'),
    'port': os.getenv('DB_PORT'),
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'sslmode': 'verify-full',
    'sslrootcert': "./certificate/RootCA.pem"
}



@contextmanager
def get_connection():
    conn = None
    try:
        logger.info(f"connecting to {conn_data}")
        conn = psycopg2.connect(**conn_data)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def generate_embedding(text):
    try:
        response = ollama.embed(
            model="nomic-embed-text",
            input=text
        )
        return response["embeddings"][0]
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise

def update_unprocessed():
    try:
        with psycopg2.connect(**conn_data) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, text FROM publication WHERE embedding IS NULL;")
                rows = cur.fetchall()
                if not rows:
                    logger.info(f"No articles needing embeddings were found")
                processed_count = 0
                total_count = len(rows)
                for row in rows:
                    id, text = row
                    if not text.strip():
                        logger.warning(f"Skipping publication ID={id} with empty text")
                        continue
                    try:
                        logger.info(f"Gennerating embedding for publication ID={id}")
                        start = time.time()
                        embedding = generate_embedding(text)

                        cur.execute(
                            "UPDATE publication SET embedding = %s WHERE id = %s",
                            (embedding, id)
                        )

                        conn.commit()
                        processed_count += 1
                        duration = time.time() - start
                        logger.info(f"Successfully processed publication ID={id} ({processed_count}/{total_count}) took {duration:.4f} sec")
                    except psycopg2.Error as e:
                        logger.error(f"Aborting! Database error occure while processing publication ID={id}: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Failed to process publication ID={id}: {e}")
                        continue


    except Exception as e:
        logger.error(f"DB connection error {e}")


if __name__ == "__main__":
    ollama.pull('nomic-embed-text')
    logger.info(f"Starting worker!")
    update_unprocessed()


