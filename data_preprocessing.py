import json
from pymongo import MongoClient as mc

import pyspark
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

client = mc('mongodb://10.100.54.129:27017/')
db = client['PaperAPI']
collection = db['paper01']

# Spark 세션 초기화
print(pyspark.__version__)
spark = SparkSession.builder \
    .appName("Word2VecExample") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("INFO")  # 로깅 수준을 INFO로 설정
fields = ["journal_name", "publisher_name", "pub_year", "article_categories", "authors"]
schema = StructType([
    StructField("journal_name", StringType(), True),
    StructField("publisher_name", StringType(), True),
    StructField("pub_year", StringType(), True),
    StructField("article_categories", StringType(), True),
    StructField("authors", ArrayType(StringType()), True)
])


def extract_authors(author_data):

    if isinstance(author_data, list):
        return [author.get('#text', '') if isinstance(author, dict) else author for author in author_data]

    elif isinstance(author_data, dict):
        return [author_data.get('#text', '')]

    elif isinstance(author_data, str):
        return [author_data]

    else:
        return []


def verify_all_fields(document_all_fields):
    if not all(value for value in document_all_fields.values()):
        return None


def extract_data(document_data):

    extract_document_field = {}

    # extract journalInfo field
    journal_info = document_data.get('journalInfo', {})
    extract_document_field['journal_name'] = journal_info.get('journal-name', '')
    extract_document_field['publisher_name'] = journal_info.get('publisher-name', '')
    extract_document_field['pub_year'] = journal_info.get('pub-year', '')

    # extract articleInfo field
    article_info = document_data.get('articleInfo', {})
    if article_info:
        if article_info.get('article-categories', '') is not None:
            extract_document_field['article_categories'] = article_info.get('article-categories', '')
        else:
            return None
    else:
        return None

    # extract author-group field in articleInfo field
    author_group = article_info.get('author-group', {})
    if author_group:
        authors = author_group.get('author', None)
        if authors is not None:
            extract_document_field['authors'] = extract_authors(authors)
    else:
        return None

    # Verify that all fields were extracted properly
    if not verify_all_fields(extract_document_field):
        return extract_document_field
    else:
        return None


def main():
    with open("data.txt", "w", encoding='utf-8') as f:
        extracted_data = []
        document_cnt = 0

        for document in collection.find():
            data = extract_data(document)
            if data:
                extracted_data.append(data)
                document_cnt += 1
                print(document_cnt, data)
                data_string = json.dumps(data, ensure_ascii=False)
                f.write(data_string + '\n')

        df = spark.createDataFrame([Row(**x) for x in extracted_data], schema=schema)
        split_words_udf = udf(lambda x: x.split(), ArrayType(StringType()))
        df = df.withColumn("authors", split_words_udf(col("authors")))

        # Word2Vec 모델 설정 및 학습
        word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="authors", outputCol="authors_emb")
        model = word2Vec.fit(df)

        # 임베딩 결과를 DataFrame에 추가
        result_df = model.transform(df)
        result_df.show()
        spark.stop()


if __name__ == "__main__":
    main()