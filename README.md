# Ukrainian Text Generation with LSTM (Keras)

**Task type:** Text generation (українською) на базі **рекурентних мереж (LSTM)**.  
**Дані:** по замовчуванню — файл `data/lys_mykyta.txt` (І. Франко “Лис Микита”, редакція М. Рильського) (додано з вашого аплоаду).  
Можна замінити на власний корпус (наприклад, датасет з Kaggle: *Ukrainian Texts*).

## Швидкий старт

```bash
# (Рекомендовано) створіть та активуйте віртуальне середовище
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt

# (Одноразово) ініціалізувати NLTK токенізатори (не обов'язково для char-level)
python -c "import nltk; nltk.download('punkt')"

# Навчання
python train_char_rnn.py --data_path data/lys_mykyta.txt --seq_len 120 --batch 128 --epochs 20

# Генерація
python generate.py --checkpoint checkpoints/best.keras --seed "Надійшла весна" --length 800 --temperature 0.8
```

## Структура
```
uk_textgen_rnn/
 ├── data/
 │   └── lys_mykyta.txt           # вихідний корпус (замінимо на ваш власний за потреби)
 ├── checkpoints/                 # збережені ваги моделі (створюється автоматично)
 ├── train_char_rnn.py            # навчання LSTM-моделі
 ├── generate.py                  # генерація тексту з чекпойнта
 ├── sampling.py                  # температурне/Top-k/Top-p семплювання
 ├── utils.py                     # допоміжні функції
 └── requirements.txt
```

## Нотатки
- Модель **char-level**: проста, стійка на малих корпусах, природно працює для української без складних токенізаторів.
- Для більших корпусів перейдіть на **word-piece/BPE + Transformer** (див. нижче *Дорожня карта*).
- Використані бібліотеки з переліку завдання: `tensorflow/keras`, `nltk` (опційно), `spacy` (опційно), а також `heapq` всередині семплера.

## Дорожня карта (за бажанням)
- **Word-level LSTM**: замінити char-векторизацію на токенізацію словами (`nltk` або `spacy`).
- **Transformer language model**: `TextVectorization` + невеликий Transformer для next-token prediction (швидший конвергент на великому корпусі).
- **Machine Translation (EN↔UK)**: використати ManyThings/Tatoeba пари (`uk.txt`/`uk-en.txt`) у форматі `src\ttrg` і навчити seq2seq (encoder–decoder з увагою).

## Ліцензія тексту
Використовуйте відкриті/дозволені корпуси та дотримуйтесь умов їхніх ліцензій.

## Запуск через єдину точку входу `main.py`

Тепер усе запускається однією командою з підкомандами **train** та **generate**:

```bash
# тренування
python main.py train --data_path data/lys_mykyta.txt --seq_len 120 --batch 128 --epochs 20

# генерація
python main.py generate --checkpoint checkpoints/best.keras --seed "Надійшла весна" --length 800 --temperature 0.8

# Генерація зображень
python main.py plot-history --history checkpoints/history.json --out img/training_curves.png

```

> --seq_len (у train) — це довжина вікна навчання. Ми ріжемо текст на шматочки по seq_len+1 символів і вчимося передбачати наступний символ з попередніх seq_len. Більший seq_len дає моделі шанс ловити довші залежності, але їсть пам’ять і час. Для char-level українською зазвичай достатньо 100–200.

> --length (у generate) — це скільки символів згенерувати поверх твого --seed. Наприклад, --length 800 = виведи 800 символів продовження.

## Практичні поради:

- Якщо текст «забуває» структуру на довгих відстанях — спробуй збільшити seq_len (скажімо, до 160–200) і трохи зменшити batch, щоб влізло в пам’ять.

- Якщо генерація здається занадто випадковою — зменшуй --temperature або увімкни --top_k 40 чи --top_p 0.9.

- --length став за потребою: 400–800 символів — норм для короткої демонстрації; для великих уривків роби 1500+.

## Резюме:

- seq_len — «скільки минулих символів ми показуємо моделі під час навчання/кроку передбачення».

- length — «скільки нових символів ти хочеш отримати на виході».
