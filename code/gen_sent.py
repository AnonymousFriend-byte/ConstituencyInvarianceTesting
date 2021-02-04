import argparse
import sys
from imp import reload
import pandas as pd
import re
import time
import string
import transformers

reload(sys)
pattern_punc = re.compile(r"^ , | ,  , | ,  .+$")
pattern_mask = re.compile(r'\[\d\]')
pattern_char = re.compile(r"^[a-zA-Z0-9-,.]+$")
pattern_1 = re.compile(r" ,|, |,")
pattern_2 = re.compile(r" - ")
phrase_list = []
punc = string.punctuation
unmasker = transformers.pipeline('fill-mask', model='bert-base-uncased')
BERT_SCORE = 0.1


def get_mask_sent(temp, i, idx, path):
    mask_phrase = pd.read_csv(path + "_mask_phrase.csv")
    print(mask_phrase.iat[i, idx + 1])
    pred_phrase = mask_phrase.iat[i, idx + 1].split(";")
    pred_list = []
    for p in pred_phrase:
        new_sent = temp.replace("[" + str(idx) + "]", p)
        new_sent = re.sub(pattern_mask, "", new_sent)
        # new_sent = re.sub(pattern_punc, "", new_sent).rstrip()
        new_sent = sent_format(new_sent)
        if new_sent[-1] != '.':
            new_sent = new_sent + '.'
        new_sent = new_sent.replace(",,",",")
        pred_list.append(new_sent)
    return pred_list


def format_sent(sent):
    punc = [",", "\"", "\$", "\(", "\)", "\.","\:"]
    for p in punc:
        idx = re.finditer(p, sent)
        count = 0
        for i in idx:
            start = i.start()
            if sent[start + count - 1] != " ":
                sent = " ".join([sent[0:start + count], sent[start + count:len(sent)]])
                count += 1
            if start + count + 1 != len(sent):
                if sent[start + count + 1] != " ":
                    sent = " ".join([sent[0:start + count + 1], sent[start + count + 1:len(sent)]])
                    count += 1
    sent = re.sub(pattern_2, "-", sent)
    idx = sent.find("[")
    sent = sent[:idx] + " " + sent[idx:]
    idx = sent.find("]")
    sent = sent[:idx + 1] + " " + sent[idx + 1:]
    idx = sent.find("'s")
    sent = sent[:idx] + " " + sent[idx:]
    if idx == -1:
        idx = sent.find("'")
        sent = sent[:idx] + " " + sent[idx:]
        idx = sent.find("'")
        sent = sent[:idx + 1] + " " + sent[idx + 1:]
    sent = " ".join(sent.split())
    return sent


def format_abbr(sent):
    abbr = ["n't", "'s", "'re", "'ll", "'m"]
    words = sent.split(" ")
    for w in words:
        if w in abbr:
            idx = sent.find(w)
            sent = sent[:idx - 1] + sent[idx:]
    return sent


def check_punc(orginal, pred):
    pred = format_sent(pred)
    orginal = format_sent(orginal)
    o_words = orginal.split(" ")
    pred_words = pred.split(" ")
    idx = o_words.index("[MASK]")
    if len(o_words) != len(pred_words):
        return False," ".join(o_words)
    if pred_words[idx] in punc:
        return False, " ".join(o_words)
    o_words[idx] = pred_words[idx]
    return True, " ".join(o_words)


def read_word_list(path):
    data = pd.read_csv(path + '_mask_word.csv')
    data = data.fillna(" ")
    row = data.shape[0]
    col = data.shape[1]
    word_list = []
    for i in range(0, row):
        words = []
        j = 1
        while data.iat[i, j] != " ":
            print(data.iat[i, j])
            words.append(data.iat[i, j])
            j += 1
            if j == col:
                break
        word_list.append(words)
    return word_list


def pred_sent(mask_sent, words):
    sent_set = set()
    mask_word = words.split(";")
    i = 0
    flag = 0
    new_sent = ""
    for sent in mask_sent:
        if "[MASK]" not in sent:
            flag = 1
            new_sent = sent
            break
        sent = re.sub(pattern_2, "-", sent)
        ans = unmasker(sent)
        for r in ans:
            if len(sent_set) > 5:
                break
            if r['score'] > BERT_SCORE:
                new_sent = r['sequence'][5:len(r['sequence']) - 5]
                result = check_punc(sent, new_sent)
                if result[0]:
                    new_sent = format_sent(result[1])
                    new_sent = format_abbr(new_sent)
                    sent_set.add(new_sent)
            else:
                break
        word = mask_word[i]
        new_sent = sent.replace("[MASK]", word)
        new_sent = format_sent(new_sent)
        new_sent = format_abbr(new_sent)
        sent_set.add(new_sent)
        i += 1
    if flag == 1:
        new_sent = format_sent(new_sent)
        new_sent = format_abbr(new_sent)
        sent_set.add(new_sent)
    return sent_set


def sent_format(sent):
    p1 = re.compile(r" [!?',;]|[!?',;] | [!?',;]| \.$| \. $|\. $")
    p2 = re.compile(r"[!?',;.]")
    res = re.findall(p1, sent)
    res_idx = re.finditer(p1, sent)
    while len(res) != 0:
        diff_v = 0
        for r in res_idx:
            start = r.span()[0] - diff_v
            end = r.span()[1] - diff_v
            content = r.group()
            s_punc = re.findall(p2, r.group())[0]
            if s_punc == ".":
                if end == len(sent):
                    temp = len(sent)
                    sent = sent[:start] + "."
                    diff_v += temp - len(sent)
            else:
                temp = len(sent)
                sent = sent.replace(content, s_punc)
                diff_v += temp - len(sent)
        res_idx = re.finditer(p1, sent)
        res = re.findall(p1, sent)
    sent = sent.strip()
    sent = sent.rstrip()
    word = sent.split(" ")
    new_s = ""
    for w in word:
        if len(w) != 0:
            w = w.strip().rstrip()
            new_s += w + " "
    new_s = new_s[:-1]
    while ",," in new_s:
        new_s = new_s.replace(",,", ',')
    new_s = new_s.replace("..", '.')
    new_s = new_s.replace(',.', '.')
    return new_s


def get_new_temp(temp, idx, s):
    s = sent_format(s)
    s_idx = temp.find("[" + str(idx) + "]")
    find_str = re.sub(pattern_mask, "", temp[(s_idx + 3):len(temp)])
    find_str = sent_format(find_str)
    if find_str != ".":
        e_idx = s.find(find_str)
        new_temp = s.replace(s[e_idx:len(s)], temp[s_idx + 3:len(temp)])
    else:
        new_temp = s[:-1] + temp[s_idx + 3:len(temp)]
    return new_temp


def format_result(sent):
    sent = sent.replace(" , ", ", ")
    sent = sent.replace(" .", ".")
    sent = re.sub(pattern_2, "-", sent).strip().rstrip()
    return sent[:1].upper() + sent[1:]


def predict_sent(pred_file, temp_file, comp_file, path, s_id, e_id):
    sent_temp = pd.read_csv(temp_file)['temp'].values
    sent_comp = pd.read_csv(comp_file)['comp'].values
    word_list = read_word_list(path)
    with open(pred_file, mode='a') as w:
        for i in range(s_id, e_id):
            w.write("sent_id = " + str(i))
            w.write("\n")
            temp = sent_temp[i]
            idx_count = temp.count('[')
            t_idx = 0
            temp_list = []
            w.write(format_result(sent_comp[i]))
            w.write("\n")
            while t_idx < idx_count:
                w.write("add" + " [" + str(t_idx) + "]\n")
                if t_idx < 1:
                    mask_sent = get_mask_sent(temp, i, t_idx, path)
                    sent_set = pred_sent(mask_sent, word_list[i][t_idx])
                    for s in sent_set:
                        s = format_result(s)
                        w.write(s + "\n")
                        new_temp = get_new_temp(temp, t_idx, s)
                        temp_list.append(new_temp)
                    w.write("\n")
                else:
                    new_temp_list = []
                    for j in range(0, len(temp_list)):
                        mask_sent = get_mask_sent(temp_list[j], i, t_idx, path)
                        sent_set = pred_sent(mask_sent, word_list[i][t_idx])
                        for s in sent_set:
                            s = format_result(s)
                            w.write(s + "\n")
                            new_temp = get_new_temp(temp_list[j], t_idx, s)
                            new_temp_list.append(new_temp)
                        w.write("\n")
                    temp_list = list(new_temp_list)
                t_idx += 1
            w.write("FIN\n")
    w.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-name',
                        default='business_sent_1')
    parser.add_argument('--sid',
                        type=int,
                        default=0)
    parser.add_argument('--eid',
                        type=int,
                        default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    file_name = args.file_name
    s_id = args.sid
    e_id = args.eid
    print("pred param:" + file_name )
    comp_file = './comp_result/' + file_name + '_comp.csv'
    temp_file = './comp_result/' + file_name + '_temp.csv'
    pred_file = './pred_sents/' + file_name + '_pred.txt'
    start = time.perf_counter()
    predict_sent(pred_file, temp_file, comp_file, './comp_result/' + file_name, s_id, e_id)
    end = time.perf_counter()
    print('Predict Sentences Running time: %s Seconds' % (end - start))



if __name__ == '__main__':
    main()
