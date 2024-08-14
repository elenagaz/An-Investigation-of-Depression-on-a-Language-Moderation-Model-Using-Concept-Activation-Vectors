import json


class TextToJsonlConverter:
    def __init__(self, input_file):
        self.input_file = input_file
        self.seen_sentences = set()

    def process_input(self):
        with open(self.input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        output_lines = []
        for line in lines:
            line = line.strip()  # for any empty lines
            if not line:
                continue

            sentence = self.extract_sentence(line)
            if sentence in self.seen_sentences:
                raise Exception(f"Duplicate found:\nPrevious: {sentence}\nNew: {line.strip()}")
            else:
                self.seen_sentences.add(sentence)
                json_line = self.create_json_line(sentence)
                output_lines.append(json_line)

        return output_lines

    def extract_sentence(self, line):
        return line.strip()

    def create_json_line(self, sentence):
        # must have at least one label for LIME
        json_object = {
            "prompt": sentence,
            "label": "OK"
        }
        return json.dumps(json_object)

    def write_to_jsonl(self, output_file):
        output_lines = self.process_input()
        with open(output_file, 'w') as file:
            for line in output_lines:
                file.write(line + '\n')


def main():
    input_file = '../data_for_each_category/raw_files/all_depression_data.txt'
    output_file = '../data_for_each_category/converted_files/converted_depression_data.jsonl'

    try:
        print(input_file)
        converter = TextToJsonlConverter(input_file)
        converter.write_to_jsonl(output_file)
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
