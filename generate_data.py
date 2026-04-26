import json
import random
import pandas as pd
from datetime import datetime, timedelta

languages = ['English', 'Hindi', 'Telugu', 'Marathi', 'Punjabi',
             'Assamese', 'Kannada', 'Tamil', 'Bengali', 'Malayalam']
sample_texts = {
    'English': 'This is an amazing post on social media! check it out @friend https://link.com #insta',
    'Hindi': 'यह सोशल मीडिया पर एक अद्भुत पोस्ट है! इसे देखें @दोस्त https://link.com #इंस्टा',
    'Telugu': 'ఇది సోషల్ మీడియాలో అద్భుతమైన పోస్ట్! ఇక్కడ చూడండి @మిత్రుడు https://link.com #ఇన్స్టా',
    'Marathi': 'हे सोशल मीडियावर एक आश्चर्यकारक पोस्ट आहे! नक्की बघा @मित्र https://link.com #इंस्टा',
    'Tamil': 'இது சமூக ஊடகங்களில் ஒரு அற்புதமான பதிவு! பார்க்கவும் @நண்பர் https://link.com #இன்ஸ்டா',
    'Bengali': 'এটি সোশ্যাল মিডিয়ায় একটি চমৎকার পোস্ট! দেখুন @বন্ধু https://link.com #ইন্সটা',
    'Malayalam': 'ഇത് സോഷ്യൽ മീഡിയയിലെ ഒരു മികച്ച പോസ്റ്റാണ്! കാണുക @സുഹൃത്ത് https://link.com #ഇൻസ്റ്റ',
    'Kannada': 'ಇದು ಸಾಮಾಜಿಕ ಮಾಧ್ಯಮದಲ್ಲಿ ಅದ್ಭುತವಾದ ಪೋಸ್ಟ್ ಆಗಿದೆ! ನೋಡಿ @స్నేహిత https://link.com #ఇన్స్టా',
    'Punjabi': 'ਇਹ ਸੋਸ਼ਲ ਮੀਡੀਆ ਤੇ ਇੱਕ ਸ਼ਾਨਦਾਰ ਪੋਸਟ ਹੈ! ਦੇਖੋ @ਦੋਸਤ https://link.com #ਇੰਸਟਾ',
    'Assamese': 'এইটো ছ’চিয়েল মিডিয়াত এক আচৰিত পোষ্ট! চাওক @বন্ধু https://link.com #ইনষ্টা'
}

def generate_osn_data(num_samples=10000):
    data = []
    for i in range(num_samples):
        lang = random.choice(languages)
        record = {
            'user_profile': {
                'user_id': f'UID_{random.randint(100000, 999999)}',
                'user_name': f'user_{lang.lower()}_{i}',
                'account_creation': (datetime.now() - timedelta(days=random.randint(100, 2500))).strftime('%Y-%m-%d'),
                'location': f'{lang}_Region',
                'followers_count': random.randint(100, 1000000),
                'following': random.randint(50, 5000)
            },
            'content_data': {
                'host_id': f'HID_{random.randint(1000, 9999)}',
                'text_caption': sample_texts[lang] + f' #{lang} #trend',
                'time_stamp': (datetime.now() - timedelta(hours=random.randint(1, 2400))).strftime('%Y-%m-%d %H:%M:%S'),
                'language': lang
            },
            'interaction_data': {
                'likes': random.randint(10, 50000),
                'shares': random.randint(0, 10000),
                'comments': random.randint(0, 5000),
                'hashtags': ['insta', lang.lower(), 'trend']
            },
            'network_relationship': {
                'mutuals': random.randint(0, 100),
                'suggested_followers': random.randint(5, 50),
                'community': f'{lang}_Group',
                'instagram_pages': [f'Page_{random.randint(1,10)}']
            },
            'engagement_matrix': {
                'engagement_rate': round(random.uniform(0.1, 25.0), 2)
            }
        }
        data.append(record)
    return data

if __name__ == "__main__":
    dataset = generate_osn_data(num_samples=10000)
    with open('instagram_multi_lang.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print('Data generated successfully')
