### Feature Engineering

import numpy as np
import re
import string
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer


def compute_poi_email_ratio(poi_messages, all_messages):
    """ FEATURE
        given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the ratio of messages to/from that person
        that are from/to a POI
    """
    ratio = 0.
    if type(poi_messages) is int and type(all_messages) is int and poi_messages > 0 and all_messages > 0:
        ratio = float(poi_messages) / float(all_messages)
    return ratio


def compute_email_addresses_per_poi(found_pois, pois_count):
    """ FEATURE
        given a dictionary with pois and the total count,
        calculate the ratio of email addresses per poi
    """
    return np.float32(sum(found_pois.itervalues()) / (pois_count+1e-10))


def compute_poi_mention_rate(poi_count, n_pois_total):
    """ FEATURE
        given the number of found pois and the total number of pois
        return the ratio of pois mentioned
    """
    return poi_count / float(n_pois_total)


def poi_vectorizer(poi_email_dict):
    vectorizer = TfidfVectorizer(sublinear_tf=True, token_pattern=r"([a-zA-Z0-9_\.\+\-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-\.]+)")
    x_train = []
    for name in poi_email_dict:
        for e in poi_email_dict[name]:
            x_train.append(e)
    vectorizer.fit(x_train)
    return vectorizer


def read_emails(f, only_poi=False):
    stemmer = SnowballStemmer('english')
    text = []
    authors = []
    from_re = re.compile(ur'From: ([\w\._-]+@[\w\._-]+\.[a-z]{2,4})')
    for line in f:
        line = line.replace('enron_mail_20110402/', '').strip()
        try:
            email_text = ''
            email_file = open(line, 'r')
            email_file.readline()  # discard this line
            email_file.readline()  # discard this line
            author_raw = email_file.readline().replace('\n', '')
            author = from_re.sub(u'\\1', author_raw, 0).strip()
            poi = get_poi_by_email(author)
            if poi is not None:
                raw = unicode(email_file.read(), 'utf-8', errors='ignore')
                #print raw.split('X-FileName:')
                email_text += raw.split('X-FileName:')[1]
                email_text = ' '.join([stemmer.stem(w) for w in email_text.split()])
                text.append(email_text)
                authors.append(poi)
            elif not only_poi:
                email_text = ' '.join([stemmer.stem(w) for w in email_text.split()])
                text.append(email_text)
                authors.append(author)
        except UnicodeDecodeError:
            pass
    return text, authors


def get_poi_by_email(email_address):
    for key in poi_email_dict:
        if email_address in poi_email_dict[key]:
            return key
    return None

def get_email_data_from_pois(dataset):
    email_texts = []
    email_authors = []
    for name in dataset:
        if dataset[name]['poi'] is True:
            email_text = ''
            data_point = dataset[name]
            try:
                from_file = open('emails_by_address/from_%s.txt' % data_point['email_address'], 'r')
                to_file = open('emails_by_address/to_%s.txt' % data_point['email_address'], 'r')
                e, a = read_emails(from_file, only_poi=True)
                email_texts.extend(e)
                email_authors.extend(a)
                e, a = read_emails(to_file, only_poi=True)
                email_texts.extend(e)
                email_authors.extend(a)
            except IOError:
                pass
    return email_texts, email_authors


def find_pois_in_data_point(data_point, vectorizer, poi_email_dict):
    found_pois = defaultdict(int)
    email_text = ''
    try:
        from_file = open('emails_by_address/from_%s.txt' % data_point['email_address'], 'r')
        to_file = open('emails_by_address/to_%s.txt' % data_point['email_address'], 'r')
        email_texts, _ = read_emails(from_file)
        email_text += ' '.join(email_texts)
        email_texts, _ = read_emails(to_file)
        email_text += ' '.join(email_texts)
        transformed = vectorizer.transform([email_text])
        emails_found = vectorizer.inverse_transform(transformed)
        for e in emails_found[0]:
            for name in poi_email_dict:
                if e in poi_email_dict[name]:
                    found_pois[name] += 1
    except IOError:
        pass
    return found_pois, len(found_pois)


def normalise_text(text):
    """ Obsolete """
    return text.translate(string.maketrans("", ""), string.punctuation).lower()


def find_poi_in_email(f, poi_names, poi_emails):
    """ Find email or name in email body
        Return True if a name or email from POI is found
        Obsolete
    """
    f.seek(0)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    if len(content) > 1:
        body_norm = normalise_text(content[1])
        for name in poi_names:
            name_v1 = normalise_text(name)
            name_v2 = normalise_text(' '.join(reversed(name.split())))
            if name_v1 in body_norm or name_v2 in body_norm:
                return True
        for email in poi_emails:
            if email in content[1]:
                return True
    return False


def scale_feature_df(df, feature, nan_strategy='median'):
    if nan_strategy is not None:
        imp = Imputer(missing_values='NaN', strategy=nan_strategy, axis=0)
    scaler = MinMaxScaler()
    all_values = df[feature].values.reshape((-1,1))
    if nan_strategy is not None:
        all_values = imp.fit_transform(all_values)
    scaled = scaler.fit_transform(all_values)
    return scaled


def scale_feature(dataset, feature, nan_strategy='median'):
    """
        Create a MinMaxScaler instance and
        scale all values of a particular feature in 
        the given dataset.
        Use an Imputer instance if nan_strategy is not None
        Return scaled values and scaler object
    """
    if nan_strategy is not None:
        imp = Imputer(missing_values='NaN', strategy=nan_strategy, axis=0)
    scaler = MinMaxScaler()
    all_values = []
    for name in dataset:
        all_values.append(dataset[name][feature])
    all_values = np.array(all_values).astype(np.float64)
    all_values = all_values.reshape((-1,1))
    if nan_strategy is not None:
        all_values = imp.fit_transform(all_values)
    scaled = scaler.fit_transform(all_values)
    return scaled, scaler


poi_names = [
	'Lay, Kenneth',
	'Skilling, Jeffrey',
	'Howard, Kevin',
	'Krautz, Michael',
	'Yeager, Scott',
	'Hirko, Joseph',
	'Shelby, Rex',
	'Bermingham, David',
	'Darby, Giles',
	'Mulgrew, Gary',
	'Bayley, Daniel',
	'Brown, James',
	'Furst, Robert',
	'Fuhs, William',
	'Causey, Richard',
	'Calger, Christopher',
	'DeSpain, Timothy',
	'Hannon, Kevin',
	'Koenig, Mark',
	'Forney, John',
	'Rice, Kenneth',
	'Rieker, Paula',
	'Fastow, Lea',
	'Fastow, Andrew',
	'Delainey, David',
	'Glisan, Ben',
	'Richter, Jeffrey',
	'Lawyer, Larry',
	'Belden, Timothy',
	'Kopper, Michael',
	'Duncan, David',
	'Bowen, Raymond',
	'Colwell, Wesley',
	'Boyle, Dan',
	'Loehr, Christopher'
]
poi_emails = [ 
    "chairman.ken@enron.com",  # not in dataset
    "kevin_a_howard.enronxgate.enron@enron.net",  # not in dataset
    "kevin.howard@enron.com",  # not in dataset
    "kevin.howard@enron.net",  # not in dataset
    "kevin.howard@gcm.com",  # not in dataset
    "michael.krautz@enron.com",  # not in dataset
    "jbrown@enron.com",  # not in dataset
    "james.brown@enron.com",  # not in dataset
    "tim_despain.enronxgate.enron@enron.net",  # not in dataset
    "tim.despain@enron.com",  # not in dataset
    "m..forney@enron.com",  # not in dataset
    "jeff.richter@enron.com",  # not in dataset
    "jrichter@nwlink.com",  # not in dataset
    "lawrencelawyer@aol.com",  # not in dataset
    "lawyer'.'larry@enron.com",  # not in dataset
    "larry_lawyer@enron.com",  # not in dataset
    "llawyer@enron.com",  # not in dataset
    "larry.lawyer@enron.com",  # not in dataset
    "lawrence.lawyer@enron.com",  # not in dataset
    "dave.duncan@enron.com",  # not in dataset
    "dave.duncan@cipco.org",  # not in dataset
    "duncan.dave@enron.com",  # not in dataset
    "dan.boyle@enron.com",  # not in dataset
    "cloehr@enron.com",  # not in dataset
    "chris.loehr@enron.com"  # not in dataset
]

poi_email_dict = {
    'LAY KENNETH L': [
        "kenneth_lay@enron.net",
        "kenneth_lay@enron.com",
        "klay.enron@enron.com",
        "kenneth.lay@enron.com",
        "klay@enron.com",
        "layk@enron.com"
    ],
    'SKILLING JEFFREY K': [
        "jeffreyskilling@yahoo.com",
        "jeff_skilling@enron.com",
        "jskilling@enron.com",
        "effrey.skilling@enron.com",
        "skilling@enron.com",
        "jeffrey.k.skilling@enron.com",
        "jeff.skilling@enron.com"
    ],
    'HANNON KEVIN P': [
        "kevin_hannon@enron.com", 
        "kevin'.'hannon@enron.com", 
        "kevin_hannon@enron.net", 
        "kevin.hannon@enron.com"
    ],
    'COLWELL WESLEY': [
        "wes.colwell@enron.com"
    ],
    'RIEKER PAULA H': [
        "paula.rieker@enron.com",
        "prieker@enron.com"
    ],
    'KOPPER MICHAEL J': [
        "michael.kopper@enron.com"
    ],
    'SHELBY REX': [
        "rex.shelby@enron.com", 
        "rex.shelby@enron.nt", 
        "rex_shelby@enron.net"
    ],
    'DELAINEY DAVID W': [
        "dave.delainey@enron.com",
        "david.w.delainey@enron.com", 
        "delainey.dave@enron.com", 
        "delainey@enron.com",         # removed '
        "david.delainey@enron.com", 
        "david.delainey'@enron.com",  # removed ' 
        "delainey'.'david@enron.com"
    ],
    'BOWEN JR RAYMOND M': [
        "ray.bowen@enron.com", 
        "raymond.bowen@enron.com", 
        "bowen@enron.com"  # removed '
    ],
    'BELDEN TIMOTHY N': [
        "tbelden@enron.com", 
        "tim.belden@enron.com", 
        "tim_belden@pgn.com", 
        "tbelden@ect.enron.com"
    ],
    'FASTOW ANDREW S': [
        "andrew.fastow@enron.com",
        "lfastow@pdq.net",  
        "andrew.s.fastow@enron.com", 
        "lfastow@pop.pdq.net", 
        "andy.fastow@enron.com",
    ],
    'CALGER CHRISTOPHER F': [
        "calger@enron.com",
        "chris.calger@enron.com", 
        "christopher.calger@enron.com", 
        "ccalger@enron.com"
    ],
    'RICE KENNETH D': [
        "ken.rice@enron.com",
        "ken_rice@enron.com", 
        "ken_rice@enron.net"
    ],
    'YEAGER F SCOTT': [
        "scott.yeager@enron.com",
        "syeager@fyi-net.com",
        "scott_yeager@enron.net",
        "syeager@flash.net"
    ],
    'HIRKO JOSEPH': [
        "joe'.'hirko@enron.com", 
        "joe.hirko@enron.com"
    ],
    'KOENIG MARK E': [
        "mkoenig@enron.com", 
        "mark.koenig@enron.com"
    ],
    'CAUSEY RICHARD A': [
        "rick.causey@enron.com", 
        "richard.causey@enron.com", 
        "rcausey@enron.com"
    ],
    'GLISAN JR BEN F': [
        "ben.glisan@enron.com", 
        "bglisan@enron.com", 
        "ben_f_glisan@enron.com", 
        "ben.glisan@enron.com"  # removed '
    ]
}