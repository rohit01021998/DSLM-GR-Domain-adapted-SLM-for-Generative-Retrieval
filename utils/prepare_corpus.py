import json
import os

def main():
    chunks_file = 'data/raw/chunks.json'
    corpus_file = 'data/raw/corpus.txt'
    delimiter = '\n\n<---CHUNK_BOUNDARY--->\n\n'

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    corpus_text = []
    for chunk in chunks:
        # Include the content
        corpus_text.append(chunk['page_content'].strip())

    # Save to corpus.txt
    os.makedirs('data/raw', exist_ok=True)
    with open(corpus_file, 'w', encoding='utf-8') as f:
        f.write(delimiter.join(corpus_text))

    print(f"Generated {corpus_file} from {len(chunks)} chunks.")

if __name__ == '__main__':
    main()
