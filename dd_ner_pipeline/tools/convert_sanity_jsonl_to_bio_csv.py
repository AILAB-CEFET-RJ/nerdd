import argparse
import csv
import json

from nltk.tokenize import TreebankWordTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Convert sanity JSONL (text+spans) to BIO CSV")
    parser.add_argument("--input-jsonl", default="../data/gliner_teste_sanidade.json")
    parser.add_argument("--output-csv", default="../data/gliner_teste_sanidade_bio.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = TreebankWordTokenizer()
    sentence_id = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as infile, open(
        args.output_csv, "w", newline="", encoding="utf-8"
    ) as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["sentence_id", "word", "label"])

        for line in infile:
            if not line.strip():
                continue

            data = json.loads(line)
            text = data["text"]
            spans = data.get("spans", [])
            tokens = tokenizer.tokenize(text)
            token_spans = list(tokenizer.span_tokenize(text))

            for token, (start, end) in zip(tokens, token_spans):
                label = "O"
                for span in spans:
                    if start >= span["start"] and end <= span["end"]:
                        tag = "B" if start == span["start"] else "I"
                        ent_type = str(span["label"]).upper()[:3]
                        label = f"{tag}-{ent_type}"
                        break
                writer.writerow([sentence_id, token, label])
            sentence_id += 1

    print(f"Conversion completed: {args.output_csv}")


if __name__ == "__main__":
    main()
