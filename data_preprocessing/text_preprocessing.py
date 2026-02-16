import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from collections import Counter
from razdel import sentenize
from datetime import datetime
import warnings
import re
from dotenv import load_dotenv
from dateparser.search import search_dates
from tqdm import tqdm
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc

warnings.filterwarnings('ignore')

load_dotenv()

MAIN_PATH = os.getenv('MAIN_PATH')


NEWS_INPUT_PATH = os.path.join(MAIN_PATH, 'data/processed/prep_extended_news_data_noeng.csv.gz')
NEWS_OUTPUT_PATH = os.path.join(MAIN_PATH, 'data/processed/prep_extended_news_data.csv.gz')

russian_stopwords = set(stopwords.words('russian'))
def apply_regex(text, 
                pattern, 
                re_method='sub', 
                flags=0, 
                unique_only=False, 
                return_as_list=True,
                try_float=False, 
                value_instead_not_found=None, 
                value_instead_one_found=None,
                value_instead_multi_found=None, 
                indexes_of_multi_found=None, 
                str_borders_like="",
                str_sep_like=" + ", 
                replacement=''):
        '''
        Applies a regular expression to the text with flexible return options
        '''
        result = None

        # apply the regular expression based on the method
        if re_method == 'findall':
            result = re.findall(pattern, text, flags=flags)
        elif re_method == 'search':
            match = re.search(pattern, text, flags=flags)
            result = match.group() if match else []
        elif re_method == 'fullmatch':
            match = re.fullmatch(pattern, text, flags=flags)
            result = match.group() if match else []
        elif re_method == 'finditer':
            result = [match.group() for match in re.finditer(pattern, text, flags=flags)]
        elif re_method == 'split':
            result = re.split(pattern, text, flags=flags)
        elif re_method == 'sub':
            result = re.sub(pattern, replacement, text, flags=flags)
        else:
            print(f"Unsupported re method: {re_method}")
            return None
        
        # if unique_only is set to True, ensure the result list contains unique items
        if unique_only and isinstance(result, list):
            result = list(dict.fromkeys(result))

        # handle cases where result isn`t returned as a list
        if not return_as_list:
            # if no match is found, return the specified value or a default one
            if len(result) == 0:
                return value_instead_not_found if value_instead_not_found is not None else (float(0) if try_float else '0')
            
            # if only one match is found, return the specified value or the match itself
            elif len(result) == 1:
                return value_instead_one_found if value_instead_one_found is not None else (float(result[0]) if try_float else result[0])
            
            # if multiple matches are found, handle based on the provided options
            elif len(result) > 1:
                if value_instead_multi_found is not None:
                    return value_instead_multi_found
                elif indexes_of_multi_found is not None:
                    if isinstance(indexes_of_multi_found, int):
                        return result[indexes_of_multi_found]
                    elif isinstance(indexes_of_multi_found, list):
                        return [result[i] for i in indexes_of_multi_found if i < len(result)]
                return str_borders_like + str_sep_like.join(result) + str_borders_like

        # handle cases where no matches are found and return value is specified
        if len(result) == 0 and value_instead_not_found is not None:
            return value_instead_not_found
        # handle cases with multiple matches and specific indexes requested
        if len(result) > 1 and indexes_of_multi_found is not None:
            if isinstance(indexes_of_multi_found, int):
                return [result[indexes_of_multi_found]]
            elif isinstance(indexes_of_multi_found, list):
                return [result[i] for i in indexes_of_multi_found if i < len(result)]
            
        # convert the result to float if requested
        if try_float:
            try:
                return [float(x) for x in result]
            except ValueError:
                pass

        return result

def random_sample(df, 
                  target_size, 
                  group_col):
  
    sources = df[group_col].value_counts()
    
    # calculate the sample size for each channel, proportional to their number in the dataset
    total_count = len(df)
    sample_sizes = (sources / total_count * target_size).round().astype(int)
    
    # create an empty df to store the samples
    sampled_df = pd.DataFrame()
    
    # create sampling for each channel
    for source, sample_size in sample_sizes.items():
        sampled_group = df[df[group_col] == source].sample(n=sample_size, random_state=42)
        sampled_df = pd.concat([sampled_df, sampled_group], axis=0)
    
    # if the total sample size is less than or greater than the target size, adjust by random rows
    if len(sampled_df) > target_size:
        sampled_df = sampled_df.sample(n=target_size, random_state=42)
    elif len(sampled_df) < target_size:
        additional_rows = df.sample(n=target_size - len(sampled_df), random_state=42)
        sampled_df = pd.concat([sampled_df, additional_rows], axis=0)
    
    return sampled_df

class TextPreprocess:
    def __init__(self, 
                 df, 
                 column,
                 disable_tqdm=False):
        self.df = df
        self.column = column
        self.disable_tqdm = disable_tqdm
        
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.embedding = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.embedding)

    def clean_column(self, 
                     new_column='cleaned_text_column'):
        '''
        Cleans the text in the specified column from unwanted symbols.
        The results are stored in a new column
        '''
        # create a new column
        self.df[new_column] = None

        # add tqdm to track progress
        for index, text in tqdm(self.df[self.column].items(), 
                                desc="Cleaning rows from unwanted symnols", 
                                unit="row",
                                disable=self.disable_tqdm):
            
            # del html tags
            text = apply_regex(text, 
                                    r'<.*?>', 
                                    re_method='sub', 
                                    replacement='')

            text = apply_regex(text, 
                                    r'[\u200b\u200c\u200d\uFEFF]', 
                                    re_method='sub', 
                                    replacement='')  
            
            text = apply_regex(text, 
                                r'&nbsp;', 
                                re_method='sub', 
                                replacement=' ')
            
            text = apply_regex(text, 
                                r'[A-Za-z0-9_-]{25,}|[qpzr0-9ac-hj-np-z]{25,}', 
                                re_method='sub', 
                                replacement=' ')

            # del emojies
            emoji_pattern = (
                "[" 
                u"\U0001F600-\U0001F64F"  
                u"\U0001F300-\U0001F5FF"  
                u"\U0001F680-\U0001F6FF"  
                u"\U0001F1E0-\U0001F1FF" 
                u"\U00002500-\U00002BEF" 
                u"\U00002702-\U000027B0"
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u"\U00010000-\U0010ffff"
                "]+"
            )
            text = apply_regex(text,
                                    emoji_pattern,
                                    re_method='sub', 
                                    replacement=' ', 
                                    flags=re.UNICODE)

            # derl hashtags
            text = apply_regex(text,
                                    r'#\S+', 
                                    re_method='sub', 
                                    replacement=' ')

            # del emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
            stext = apply_regex(text, 
                                    email_pattern, 
                                    re_method='sub', 
                                    replacement=' ')

            # remove mentions starting with @
            text = apply_regex(text, 
                                    r'@\S+', 
                                    re_method='sub', 
                                    replacement=' ')

            # del url
            url_pattern = r'http[s]?://\S+|www\.\S+'
            text = apply_regex(text, 
                                    url_pattern, 
                                    re_method='sub', 
                                    replacement=' ')

            # del @, // 
            text = apply_regex(text, 
                                    r'[@/]', 
                                    re_method='sub', 
                                    replacement=' ')

            # fix missing space after dots 
            text = apply_regex(text, 
                                    r'([.!?:])(?!\s)([А-ЯЁ])', 
                                    re_method='sub', 
                                    replacement=r'\1 \2')
            
            # remove extra spaces and update the new column in the df
            text = apply_regex(text, 
                               r'\s+', 
                               re_method='sub', 
                               replacement=' ').strip()

            self.df.at[index, new_column] = text

        return self.df

    @staticmethod
    def split_into_sentences(text):
        '''
        Divides the text into sentences
        '''
        if not isinstance(text, str):
            text = ''
        sentences = [sentence.text for sentence in sentenize(text)]
        return sentences

    @staticmethod
    def find_frequent_sentences(df, 
                                channel_column='channel', 
                                text_column='cleaned_text_column', 
                                threshold=0.05, 
                                min_group_size=20):
        '''
        Finds frequent sentences for each channel
        '''
        result = {}
        grouped = df.groupby(channel_column)

        for channel, group in grouped:
            if len(group) <= min_group_size:
                 continue
            all_sentences = []
            for content in group[text_column]:
                sentences = TextPreprocess.split_into_sentences(content)
                all_sentences.extend(sentences)

            # count the frequency of occurrence of each sentence
            sentence_counts = Counter(all_sentences)
            num_texts = len(group)

            # filter the sentences that occur more often than in the specified frequency
            frequent_sentences = [sent for sent, count in sentence_counts.items() if count / num_texts > threshold]

            result[channel] = frequent_sentences

        return result

    @staticmethod
    def remove_frequent_sentences(text, 
                                  frequent_sentences):
        '''
        Removes frequency sentences from the text
        '''
        sentences = TextPreprocess.split_into_sentences(text)
        filtered_sentences = [sentence for sentence in sentences if sentence not in frequent_sentences]
        return ' '.join(filtered_sentences)

    def process_texts(self, 
                  df, 
                  channel_column='channel', 
                  text_column='cleaned_text_column', 
                  threshold=0.05):
        '''
        Removes frequency sentences from each text depending on the channel 
        '''
        frequent_sentences_by_channel = self.find_frequent_sentences(df, 
                                                                     channel_column=channel_column, 
                                                                     text_column=text_column, 
                                                                     threshold=threshold)
        for index, row in df.iterrows():
            channel = row[channel_column]  
            text = row[text_column] 
            frequent_sentences = frequent_sentences_by_channel.get(channel, [])
            df.at[index, text_column] = self.remove_frequent_sentences(text, 
                                                                       frequent_sentences)

        return df


    def remove_dates(self, 
                     df, 
                     content_column, 
                     date_column, 
                     output_column=None):
        '''
        Processes texts in the content column and dates that are not historical, 
        also removes numbers that are not related to dates or mentions of laws
        '''
        all_processed_texts = []
        processed_texts = []

        if output_column is None:
            output_column = content_column  

        for index, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Processing laws and dates:",
            disable=self.disable_tqdm,
        ):
            publication_date = self._parse_publication_date(row[date_column])
            processed_text = self._process_text(row[content_column], publication_date)
            processed_texts.append(processed_text)
            all_processed_texts.append(processed_text)

        df[output_column] = all_processed_texts  
        return df

    def _process_text(self, 
                      text, 
                      publication_date):
        '''
        Processes the text: dates that are not historical, preserves the laws,
        deletes numbers that are not related to dates or laws and clears the text
        '''
        current_year = datetime.now().year

        try:
            text = text.lower()

            # to store placeholers
            protected_items = {}

            # dels future dates and protect the dates
            matches = search_dates(text, 
                                   languages=['ru'], 
                                   settings={'RETURN_AS_TIMEZONE_AWARE': False,
                                             'PREFER_DAY_OF_MONTH': 'first', 
                                             'SKIP_TOKENS': ['ч', 'г', 'д', 'м', 'вс'],
                                             'RETURN_TIME_AS_PERIOD': False,
                                             'PARSERS': ['absolute-time', 'relative-time']})

        
            if matches:
                for i, match in enumerate(list(matches)):
                    matched_text, matched_date = match
                    matched_text_number = None
                    try:
                        matched_text_number = int(matched_text)
                    except ValueError:
                        pass
                    else:
                        if not (2000 < matched_text_number < 2050):
                            continue

                    matched_date = matched_date.replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
                    publication_date = publication_date.replace(tzinfo=None)

                    # if the date is non-historical and is equal to the current year, we delete it
                    if matched_date >= publication_date or matched_date.year == current_year:
                        text = text.replace(matched_text, '')
                    else:
                        # else replace it by placeholder
                        placeholder = f'__DATE_{i}__'
                        protected_items[placeholder] = matched_text
                        text = text.replace(matched_text, placeholder)

            # trying to identify mentions of laws 
            law_patterns = [r'''(
            (?:
                [№]?\s*[\dа-я\-–.]+\s*
            )?
            (?:
                закон[а-я]*|
                указ[а-я]*|
                кодекс[а-я]*|
                фз|
                гк\s*рф|
                ук\s*рф|
                коап\s*рф|
                нк\s*рф|
                упк\s*рф|
                тк\s*рф|
                постановлен[иея]|
                пункт|п\.|
                част[ья]|
                ч\.|
                стать[яи]|
                ст\.
            )
            \s*
            (?:
                [№]?\s*[\dа-я\-–.]+
            )?
            (?:
                \s*
                (?:
                    ук\s*рф|
                    гк\s*рф|
                    коап\s*рф|
                    нк\s*рф|
                    тк\s*рф|
                    фз
                )
            )?
            )''']
            for pattern in law_patterns:
                for match in re.findall(pattern, text, flags=re.IGNORECASE | re.VERBOSE):
                    placeholder = f'__LAW_{len(protected_items)}__'
                    protected_items[placeholder] = match
                    text = text.replace(match, placeholder)

            placeholder_pattern = r'__\w+__'
            parts = re.split(f'({placeholder_pattern})', text)

            processed_parts = []
            for part in parts:
                if re.match(placeholder_pattern, part):
                    processed_parts.append(part)
                else:
                    allowed_chars_pattern = r'[^а-яА-ЯёЁa-zA-Z0-9\s]'
                    part = re.sub(allowed_chars_pattern, ' ', part)

                    part = re.sub(r'\d', ' ', part)

                    # del extra spaces
                    part = re.sub(r'\s+', ' ', part).strip()
                    processed_parts.append(part)

            text = ' '.join(processed_parts)
            # return original values protected by placeholders
            for placeholder, original_text in protected_items.items():
                text = text.replace(placeholder, original_text)

            return self._clean_extra_spaces(text)
        except OverflowError:
            pass
        return self._clean_extra_spaces(text)

    @staticmethod
    def _clean_extra_spaces(text):
        text = re.sub(r'[^а-яА-ЯёЁa-zA-Z0-9\s.:]', ' ', text)
        text = re.sub(r'\s+\.(?=\S)', '.', text)
        text = re.sub(r'(?<=\D)\s+\.(?=\D)', '.', text)
        text = re.sub(r'(?<!\d)\.|[^\w\s.]|(?<!\d)\.(?!\d)', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def _parse_publication_date(pub_date):
        if isinstance(pub_date, str):
            return pd.to_datetime(pub_date, errors='coerce').tz_localize(None)
        return pub_date

    def remove_stopwords(self, 
                         text, 
                         russian_stopwords):
        '''
        Del stop-words
        '''
        words = text.split()
        filtered_text = ' '.join([word for word in words if word not in russian_stopwords])
        return filtered_text

    def lemmatize_text(self, 
                       text, 
                       russian_stopwords):
        '''
        Lemmatization
        '''
        if not isinstance(text, str):
            text = ''
        text = self.remove_stopwords(text, 
                                     russian_stopwords)
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)

        lemmatized_text = ' '.join(token.lemma for token in doc.tokens)
        return lemmatized_text

    def lemmatize_dataframe(self, 
                            df, 
                            column, 
                            russian_stopwords):
        '''
        Lemmatization of the text for the specified row
        '''
        tqdm.pandas(desc="Text lemmatization")  
        lemmatized_texts = []

        for text in tqdm(df[column], desc="row lemmatization"):
            lemmatized_texts.append(self.lemmatize_text(text, 
                                                        russian_stopwords))

        df[column] = lemmatized_texts
        return df

    @staticmethod
    def get_unique_frequent_words(df, 
                                  channel_column, 
                                  text_column, 
                                  top_n=5):
        '''
        Gets unique frequency words for each channel
        '''
        df['tokens'] = df[text_column].apply(lambda x: x.split())

        # calculate the total frequency of words across all channels
        total_counter = Counter()
        for tokens in df['tokens']:
            total_counter.update(tokens)
            
        result = {}

        channels = df[channel_column].unique()

        for channel in channels:
            # texts of the current channel
            channel_tokens = df[df[channel_column] == channel]['tokens']

            # calculate the frequency of words in the current channel
            channel_counter = Counter()
            for tokens in channel_tokens:
                channel_counter.update(tokens)

            # calculate the ratio of the frequency of the word in the channel to the total frequency of the entire dataframe (all channels)
            word_scores = {}
            for word in channel_counter:
                if total_counter[word] > 0:
                    word_scores[word] = channel_counter[word] / total_counter[word]

            # sort the words in descending order of the ratio and take the top-n
            top_words = sorted(word_scores.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:top_n]
            result[channel] = [word for word in top_words]

        return result

    @staticmethod
    def remove_words_from_text(text, 
                               words_to_remove):
        '''
        Del the specified words from the text
        '''
        return ' '.join([word for word in text.split() if word not in words_to_remove])

    def remove_frequent_words(self, 
                              df, 
                              channel_column, 
                              text_column, 
                              frequent_words, 
                              output_column):
        '''
        Del the detected frequency words in the news
        '''
        for index, row in df.iterrows():
            channel = row[channel_column]
            text = row[text_column]

            words_to_remove = frequent_words.get(channel, [])

            processed_text = self.remove_words_from_text(text, words_to_remove)
            df.at[index, output_column] = processed_text

        return df
    
    def remove_foreign_words(self, 
                             df, 
                             column, 
                             output_column=None):
   
        if output_column is None:
            output_column = column  
            
        df[output_column] = df[column].apply(lambda text: re.sub(r'[^а-яА-ЯёЁ0-9\s\.]', '', text))
        df[output_column] = df[output_column].apply(lambda text: re.sub(r'\s+', ' ', text).strip())
        
        return df
    

def full_text_processing(df, 
                         text_column, 
                         date_column, 
                         channel_column,
                         stopwords_set, 
                         top_n_words=3,
                         frequent_sentence_threshold=0.05):
    '''
   Performs full text processing in stages:
    1. Clearing the text of unnecessary characters
    2. Removing frequency sentences for each channel
    3. Removing unnecessary mentions from the text
    4. Removing stop words
    5. Lemmatization of the text
    6. Removing frequent words for each channel
    '''
    
    text_processor = TextPreprocess(df, 
                                    text_column)

    # Clearing the text of unnecessary characters
    df = text_processor.clean_column(new_column='cleaned_text_column')
    
    # Removing frequency sentences for each channel
    df = text_processor.process_texts(df, 
                                      threshold=frequent_sentence_threshold)
    
    # date removing
    df = text_processor.remove_dates(df, 
                                     content_column='cleaned_text_column', 
                                     date_column=date_column)

    # Removing stop words
    df['cleaned_text_column'] = df['cleaned_text_column'].apply(lambda text: text_processor.remove_stopwords(text, 
                                                                                                             stopwords_set))

    # remove foreign words
    df = text_processor.remove_foreign_words(df,
                                             column='cleaned_text_column')
    
    #  Lemmatization of the text
    df = text_processor.lemmatize_dataframe(df, 
                                            column='cleaned_text_column', 
                                            russian_stopwords=stopwords_set)

    frequent_words = text_processor.get_unique_frequent_words(df, 
                                                              channel_column, 
                                                              text_column='cleaned_text_column', 
                                                              top_n=top_n_words)

    # Removing frequent words for each channel
    df = text_processor.remove_frequent_words(df, 
                                              channel_column, 
                                              'cleaned_text_column', 
                                              frequent_words, 
                                              output_column='final_processed_text')

    return df

def remove_duplicates(df,
                      column):
    df_true_label_1 = df[df['true_label'] == 1].drop_duplicates(subset=column, keep='last')
    df_other = df[df['true_label'] != 1].drop_duplicates(subset=column, keep='last')
    df_other = df_other[~df_other[column].isin(df_true_label_1[column])]
    return pd.concat([df_true_label_1, df_other], ignore_index=True)

df = pd.read_csv(NEWS_INPUT_PATH, sep=';', quotechar='"', encoding='utf-8', compression='gzip')
df = remove_duplicates(df,column='url')


processed_df = full_text_processing(df, 
                                    text_column='content', 
                                    date_column='date', 
                                    channel_column='channel', 
                                    stopwords_set=russian_stopwords, 
                                    frequent_sentence_threshold=0.05, 
                                    top_n_words=3)

processed_df = processed_df[(processed_df['final_processed_text'] != '')& (processed_df['final_processed_text'].notna())]
processed_df = processed_df[['date', 'url', 'channel', 'data_source', 'true_label', 'content', 'final_processed_text']]

processed_df.to_csv(NEWS_OUTPUT_PATH, sep=';', compression='gzip', encoding='utf-8', index=False, quotechar='"')

