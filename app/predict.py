import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class_names = ['acanthosis-nigricans', 'achenbach-syndrome', 'acne', 'acral-lentiginous-melanoma', 'allergic-contact-dermatitis', 'alopecia-areata', 'amelanotic-melanoma', 'angioedema', 'angular-cheilitis', 'annular-lichenoid-dermatitis-of-youth', 'anogenital-warts', 'aquagenic-wrinkling-of-the-palms', 'atopic-dermatitis', 'atopic-hand-dermatitis', 'atypical-fibroxanthoma', 'atypical-mycobacterial-infection', 'autonomic-denervation-dermatitis', 'basal-cell-carcinoma', 'becker-naevus', 'bed-bugs', 'behcet-disease', 'calcinosis-cutis', 'catscratch-disease', 'cholinergic-urticaria', 'chronic-actinic-dermatitis', 'chronic-plaque-psoriasis', 'chronic-superficial-scaly-dermatitis', 'congenital-malalignment-of-the-great-toenails', 'congenital-melanocytic-naevi', 'contact-urticaria', 'covid-19', 'cutaneous-adverse-effects-of-anticonvulsant-drugs', 'cutaneous-squamous-cell-carcinoma', 'dengue', 'dermatitis-herpetiformis', 'discoid-eczema', 'discoid-lupus-erythematosus', 'drug-induced-photosensitivity', 'drug-induced-pigmentation', 'dyshidrotic-eczema', 'elastosis-perforans-serpiginosa', 'elephantiasis-nostras-verrucosa', 'erosive-lichen-planus', 'erosive-pustular-dermatosis', 'erythroderma', 'fixed-drug-eruption', 'folliculitis-barbae', 'fournier-gangrene', 'fungal-nail-infections', 'gardner-syndrome', 'generalised-eruptive-keratoacanthomas', 'generalised-pustular-psoriasis', 'granulomatous-cheilitis', 'habit-tic-deformity', 'hand-foot-and-mouth-disease', 'hereditary-coproporphyria', 'herpetic-whitlow', 'ichthyosis', 'idiopathic-facial-aseptic-granuloma', 'impetigo', 'kaposi-sarcoma', 'keloid-and-hypertrophic-scar', 'laugier-hunziker-syndrome', 'lentigo-maligna-and-lentigo-maligna-melanoma', 'lice', 'lichen-planus', 'lichen-simplex', 'linear-iga-bullous-disease', 'lipodystrophy', 'lipoma-and-liposarcoma', 'lobomycosis', 'maculopapular-cutaneous-mastocytosis', 'male-genital-dysaesthesia', 'marine-wounds-and-stings', 'melanocytic-naevus', 'melanoma-in-situ', 'molluscum-contagiosum', 'morbihan-disease', 'mpox', 'neutrophilic-sebaceous-adenitis', 'nodular-melanoma', 'nodular-prurigo', 'obesity-associated-lymphoedematous-mucinosis', 'oral-lichen-planus', 'otitis-externa', 'palmar-erythema', 'panton-valentine-leukocidin-staphylococcus-aureus', 'paradoxical-psoriasis', 'paraneoplastic-pemphigus', 'pemphigus-vulgaris', 'periorificial-dermatitis', 'peristomal-intestinal-metaplasia', 'phytophotodermatitis', 'pitted-keratolysis', 'pityriasis-alba', 'pityriasis-amiantacea', 'pityriasis-lichenoides', 'pityriasis-rosea', 'pityriasis-versicolor', 'plasma-cell-mucositis', 'polymorphic-light-eruption', 'pressure-ulcer', 'pretibial-myxoedema', 'pseudoxanthoma-elasticum', 'puva-photochemotherapy', 'pyoderma-gangrenosum', 'relapsing-polychondritis', 'rosacea', 'rubinstein-taybi-syndrome', 'sarcoidosis', 'scabies', 'scalp-psoriasis', 'sebaceous-naevus', 'seborrhoeic-dermatitis', 'segmental-pigmentation-disorder', 'shrimp-nail', 'skin-grafting', 'solar-elastosis', 'staphylococcal-scalded-skin-syndrome', 'steroid-rosacea', 'superficial-spreading-melanoma', 'systemic-amyloidosis', 'tinea-capitis', 'trachyonychia', 'trichoepithelioma', 'vitiligo', 'vulval-cancer', 'white-nail']

# Function to load and preprocess the image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_disease(img_path):
    # Load the trained model
    model = load_model('best_model.h5')

    # Load and preprocess the image
    img = load_and_preprocess_image(img_path)

    # Predict the class probabilities
    predictions = model.predict(img)
    pred = np.argmax(predictions)
    # accuracy = predictions[0][pred]
    # accuracy = round(accuracy*100, 2)

    max_val = 0
    max_idx = 0

    for idx, prob in enumerate(predictions[0]):
        if prob > max_val:
            max_idx = idx
            max_val = prob
    accuracy = round(max_val*100, 2)

    response = {
        "disease": class_names[max_idx],
        "probability": int(max_val),
        "accuracy": accuracy,
        "prediction": int(pred)
    }

    return response
