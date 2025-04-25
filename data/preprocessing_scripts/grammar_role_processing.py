import pandas as pd
import spacy
import re



def clean_utterance(utterance):
    '''remove brackets and gesture notations from utterance'''
    utt_clean = str(utterance).replace('[','').replace(']','')
    utt_clean=re.sub(" \{.*?\}","", utt_clean)
    return utt_clean

def extract_phrases_from_utt(utterance):
    '''extract noun phrase from uncleaned utterance'''
    #Find all phrases inside square brackets
    phrases = re.findall(r'\[.*?\]', utterance)
    for idx_phrase in range(len(phrases)): 
        # if re.findall(r'\[.*?\]', phrases[idx_phrase]):
        #     phrases.append(extract_phrases_from_utt(phrases[idx_phrase]))
        phrases[idx_phrase] = clean_utterance(phrases[idx_phrase])

    return phrases

def find_np_role(phrase, utterance, nlp_model):
    '''find noun phrase role in sentance (nsubj, dobj, pobj)'''
    np_role = False
    np_text = False
    doc_phrase = nlp_model(phrase)
    for chunk in doc_phrase.noun_chunks:
        np_text = chunk.text

    doc_utt = nlp_model(utterance)
    # spacy.displacy.serve(doc_utt, style="dep")
    for chunk in doc_utt.noun_chunks:
        # print(chunk.text)
        if chunk.text == np_text:
            np_role = chunk.root.dep_
    return np_role

def run_tests(nlp_model):
    # #Test 1: Easy
    # utterance1 = "Okay. Then make it easier. So we need [that blue cube looking one]..."
    # utterance1_phrases_answer = ["that blue cube looking one"]
    # utterance1_clean_answer  = "Okay. Then make it easier. So we need that blue cube looking one..."
    # utterance1_roles_answer = ["nsubj"]

    # utterance1_phrases = extract_phrases_from_utt(utterance1)
    # print(utterance1_phrases)
    # assert utterance1_phrases == utterance1_phrases_answer
    # # if utterance1_phrase != utterance1_phrase_answer:
    # #     print("phrase extraction test failed")

    # utterance1_clean = clean_utterance(utterance1)
    # print(utterance1_clean)
    # assert utterance1_clean == utterance1_clean_answer
    
    # utterance1_roles = []
    # for phrase in utterance1_phrases:
    #     utterance1_roles.append(find_np_role(phrase, utterance1_clean, nlp_model))
    # print(utterance1_roles)
    # assert utterance1_roles == utterance1_roles_answer

    # #Test 2: Multiple Seperate References
    # utterance = "And on top of [that], [it's] going to be [a red one]."
    # utterance_phrases_answer = ["that", "it's", "a red one"]
    # utterance_clean_answer  = "And on top of that, it's going to be a red one."
    # utterance_roles_answer = ["pobj", "nsubj", False]

    # utterance_phrases = extract_phrases_from_utt(utterance)
    # print(utterance_phrases)
    # assert utterance_phrases == utterance_phrases_answer
    # # if utterance1_phrase != utterance1_phrase_answer:
    # #     print("phrase extraction test failed")

    # utterance_clean = clean_utterance(utterance)
    # print(utterance_clean)
    # assert utterance_clean == utterance_clean_answer
    
    # utterance_roles = []
    # for phrase in utterance_phrases:
    #     utterance_roles.append(find_np_role(phrase, utterance_clean, nlp_model))
    # print(utterance_roles)
    # assert utterance_roles == utterance_roles_answer

    #Test 3: Multiple Nested References
    utterance = "you need [one more of [those]]."
    utterance_phrases_answer = ["one more of those", "those"]
    utterance_clean_answer  = "you need one more of those."
    utterance_roles_answer = ["nsubj", "pobj", False]

    utterance_phrases = extract_phrases_from_utt(utterance)
    print(utterance_phrases)
    if utterance_phrases != utterance_phrases_answer:
        print("manual assistance needded")
    # if utterance1_phrase != utterance1_phrase_answer:
    #     print("phrase extraction test failed")

    utterance_clean = clean_utterance(utterance)
    print(utterance_clean)
    assert utterance_clean == utterance_clean_answer
    
    utterance_roles = []
    for phrase in utterance_phrases:
        utterance_roles.append(find_np_role(phrase, utterance_clean, nlp_model))
    print(utterance_roles)
    assert utterance_roles == utterance_roles_answer

    #Test 4: Gestures

def add_grammatical_role(dataset_df, nlp_model):
    dataset_df['grammatical role'] = False
    previous_utt = ""
    same_utterance_as_prev = False
    contigous_utt_idx = 0
    for i, row in dataset_df.iterrows():
        same_utterance_as_prev = (row["Reference"] == previous_utt)
        if same_utterance_as_prev:
            contigous_utt_idx += 1
        else:
            contigous_utt_idx = 0

        
        try:
            utterance_phrases = extract_phrases_from_utt(row["Reference"])
            utterance_clean = clean_utterance(row["Reference"])
            role = find_np_role(utterance_phrases[contigous_utt_idx], utterance_clean, nlp_model)
            dataset_df.at[i,'grammatical role'] = role
        except:
            dataset_df.at[i,'grammatical role'] = "input manually"

        previous_utt = row["Reference"]

        # print (utterance_clean)
    #add selenium code

    return dataset_df


nlp_model = spacy.load("en_core_web_lg")

p1_df = pd.read_csv('data/Coding/P2.xlsx - Coding.csv')
p1_with_gramm = add_grammatical_role(p1_df, nlp_model)
p1_with_gramm.to_csv('data/Coding Grammatical Role/P2.xlsx - Coding (grammatical role).csv')

p1_df = pd.read_csv('data/Coding/P1.xlsx - Coding.csv')
p1_with_gramm = add_grammatical_role(p1_df, nlp_model)
p1_with_gramm.to_csv('data/Coding Grammatical Role/P1.xlsx - Coding (grammatical role).csv')


p1_df = pd.read_csv('data/Coding/P3.xlsx - Coding.csv')
p1_with_gramm = add_grammatical_role(p1_df, nlp_model)
p1_with_gramm.to_csv('data/Coding Grammatical Role/P3.xlsx - Coding (grammatical role).csv')


p1_df = pd.read_csv('data/Coding/P5.xlsx - Coding.csv')
p1_with_gramm = add_grammatical_role(p1_df, nlp_model)
p1_with_gramm.to_csv('data/Coding Grammatical Role/P5.xlsx - Coding (grammatical role).csv')


p1_df = pd.read_csv('data/Coding/P6.xlsx - Coding.csv')
p1_with_gramm = add_grammatical_role(p1_df, nlp_model)
p1_with_gramm.to_csv('data/Coding Grammatical Role/P6.xlsx - Coding (grammatical role).csv')


p1_df = pd.read_csv('data/Coding/P7.xlsx - Coding.csv')
p1_with_gramm = add_grammatical_role(p1_df, nlp_model)
p1_with_gramm.to_csv('data/Coding Grammatical Role/P7.xlsx - Coding (grammatical role).csv')


p1_df = pd.read_csv('data/Coding/P8.xlsx - Coding.csv')
p1_with_gramm = add_grammatical_role(p1_df, nlp_model)
p1_with_gramm.to_csv('data/Coding Grammatical Role/P8.xlsx - Coding (grammatical role).csv')

p1_df = pd.read_csv('data/Coding/P10.xlsx - Coding.csv')
p1_with_gramm = add_grammatical_role(p1_df, nlp_model)
p1_with_gramm.to_csv('data/Coding Grammatical Role/P10.xlsx - Coding (grammatical role).csv')


p1_df = pd.read_csv('data/Coding/P11.xlsx - Coding.csv')
p1_with_gramm = add_grammatical_role(p1_df, nlp_model)
p1_with_gramm.to_csv('data/Coding Grammatical Role/P11.xlsx - Coding (grammatical role).csv')


# run_tests(nlp)
