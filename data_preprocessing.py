from pymongo import MongoClient as mc

client = mc('mongodb://10.100.54.129:27017/')
db = client['PaperAPI']
collection = db['paper01']


def extract_data(document_data):

    extract_document_field = {}

    # extract journalInfo field
    journal_info = document_data.get('journalInfo', {})
    extract_document_field['journal_name'] = journal_info.get('journal-name', '')
    extract_document_field['publisher_name'] = journal_info.get('publisher-name', '')
    extract_document_field['pub_year'] = journal_info.get('pub-year', '')

    # extract articleInfo field
    article_info = document_data.get('articleInfo', {})
    if extract_document_field is not None:
        extract_document_field['article_categories'] = article_info.get('article-categories', '')
    else:
        return

    # extract author-group field
    author_group = article_info.get('author-group', {})
    if author_group is not None:
        authors = author_group.get('author', [])
        extract_document_field['authors'] = [author.get('#text', '') for author in authors if isinstance(author, dict)]
    else:
        return

    return extract_document_field


def main():
    document_cnt = 0
    for document in collection.find():
        extracted_data = extract_data(document)
        document_cnt += 1
        print(extracted_data)
        print(document_cnt)


if __name__ == "__main__":
    main()