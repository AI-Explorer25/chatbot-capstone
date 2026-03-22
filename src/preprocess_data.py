import os
import re

# ===============================
# Define paths
# ===============================
root_dir = os.path.dirname(os.path.abspath(__file__))  # src folder
data_dir = os.path.join(root_dir, "..", "data")

cornell_dir = os.path.join(data_dir, "cornell-movie-dialogs-corpus")
dailydialog_dir = os.path.join(data_dir, "dialogs-train")

output_file = os.path.join(data_dir, "combined_chat_data.txt")

# ===============================
# Helper functions
# ===============================
def clean_text(text):
    """
    Normalize text:
    - lowercase
    - remove extra spaces
    - fix spacing around punctuation
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # remove multiple spaces
    text = re.sub(r"\s([?.!,;:])", r"\1", text)  # fix space before punctuation
    text = text.strip()
    return text

# ===============================
# Process Cornell Movie Dialogs
# ===============================
# Load movie_lines
lines = {}
with open(os.path.join(cornell_dir, "movie_lines.txt"), encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split(" +++$+++ ")
        if len(parts) == 5:
            line_id = parts[0]
            text = clean_text(parts[4])
            lines[line_id] = text

# Load conversations
conversations = []
with open(os.path.join(cornell_dir, "movie_conversations.txt"), encoding="utf-8", errors="ignore") as f:
    for conv in f:
        parts = conv.strip().split(" +++$+++ ")
        if len(parts) == 4:
            # lines are stored as list string: "['L598485', 'L598486', ...]"
            line_ids_str = parts[3]
            line_ids = eval(line_ids_str)  # convert string list to Python list safely
            conv_lines = [lines[lid] for lid in line_ids if lid in lines]
            if len(conv_lines) > 1:
                # create response pairs: each line with the next line
                for i in range(len(conv_lines) - 1):
                    conversations.append(f"{conv_lines[i]}\t{conv_lines[i+1]}")

# ===============================
# Process DailyDialog Train Data
# ===============================
daily_file = os.path.join(dailydialog_dir, "dialogues_train.txt")
with open(daily_file, encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if line:
            # DailyDialog format uses "__eou__" to separate utterances
            utterances = [clean_text(u) for u in line.split("__eou__") if u.strip()]
            if len(utterances) > 1:
                for i in range(len(utterances) - 1):
                    conversations.append(f"{utterances[i]}\t{utterances[i+1]}")

# ===============================
# Save combined dataset
# ===============================
with open(output_file, "w", encoding="utf-8") as f:
    for conv in conversations:
        f.write(conv + "\n")

print(f"Preprocessing complete! Combined dataset saved to: {output_file}")
print(f"Total pairs: {len(conversations)}")