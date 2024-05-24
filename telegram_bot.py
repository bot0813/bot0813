# -*- coding: utf-8 -*-

import logging
import pandas as pd
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import random

# Включаем ведение журнала
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Путь к файлу с моделью и данными
MODEL_FILE = 'model.pkl'
DATA_FILE = 'data.csv'

# Загрузим данные для начального обучения (если файл существует)
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=['text', 'label'])

# Создадим модель машинного обучения
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Функция для обучения модели
def train_model():
    if len(df) > 0:
        model.fit(df['text'], df['label'])
        joblib.dump(model, MODEL_FILE)
        df.to_csv(DATA_FILE, index=False)

# Загрузка модели, если файл существует
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    train_model()

# Колода карт Таро и их значения
tarot_cards = {
    "The Fool": "Новый старт, наивность, чистота, спонтанность.",
    "The Magician": "Мастерство, адаптация, уверенность, сила воли.",
    "The High Priestess": "Интуиция, мудрость, тайны, подсознание.",
    "The Empress": "Творчество, природа, изобилие, материнство.",
    "The Emperor": "Власть, структура, стабильность, лидерство.",
    "The Hierophant": "Традиции, духовность, обучение, совет.",
    "The Lovers": "Любовь, отношения, выбор, гармония.",
    "The Chariot": "Решительность, победа, контроль, амбиции.",
    "Strength": "Сила, мужество, терпение, стойкость.",
    "The Hermit": "Интроспекция, одиночество, мудрость, поиск.",
    "Wheel of Fortune": "Удача, судьба, цикл, изменения.",
    "Justice": "Справедливость, истина, закон, равновесие.",
    "The Hanged Man": "Подвешенное состояние, жертва, пересмотр, отпускание.",
    "Death": "Конец, трансформация, переход, изменение.",
    "Temperance": "Умеренность, баланс, исцеление, терпение.",
    "The Devil": "Иллюзии, ограничения, привязанности, материальность.",
    "The Tower": "Крушение, потрясение, разрушение, откровение.",
    "The Star": "Надежда, вдохновение, мир, спокойствие.",
    "The Moon": "Иллюзии, страхи, сны, подсознание.",
    "The Sun": "Радость, успех, ясность, позитив.",
    "Judgement": "Суд, обновление, призыв, пробуждение.",
    "The World": "Завершение, достижение, целостность, путешествие."
}

# Обработчики команд
def start(update: Update, _: CallbackContext) -> None:
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Привет, {user.mention_markdown_v2()}\! Я обучающийся бот\.',
        reply_markup=ForceReply(selective=True),
    )

def help_command(update: Update, _: CallbackContext) -> None:
    update.message.reply_text(
        'Используйте /train <текст> <метка> для обучения или просто отправьте текст для предсказания.\n'
        'Доступные команды:\n'
        '/train <текст> <метка> - обучить модель\n'
        '/labels - показать все метки\n'
        '/accuracy - показать точность модели\n'
        '/delete_data - удалить все данные\n'
        '/log - показать журнал предсказаний\n'
        '/tarot - получить предсказание на картах Таро'
    )

def train(update: Update, context: CallbackContext) -> None:
    try:
        text = ' '.join(context.args[:-1])
        label = context.args[-1]
        global df
        df = df.append({'text': text, 'label': label}, ignore_index=True)
        train_model()
        update.message.reply_text('Модель успешно обучена на новом примере.')
    except IndexError:
        update.message.reply_text('Пожалуйста, предоставьте текст и метку в формате: /train <текст> <метка>')

def predict(update: Update, _: CallbackContext) -> None:
    text = update.message.text
    prediction = model.predict([text])
    update.message.reply_text(f'Предсказанная метка: {prediction[0]}')
    # Журналирование предсказаний
    with open('predictions.log', 'a') as log_file:
        log_file.write(f'{text} -> {prediction[0]}\n')

def show_labels(update: Update, _: CallbackContext) -> None:
    labels = df['label'].unique()
    update.message.reply_text(f'Доступные метки: {", ".join(labels)}')

def show_accuracy(update: Update, _: CallbackContext) -> None:
    if len(df) < 2:
        update.message.reply_text('Недостаточно данных для оценки точности.')
    else:
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        update.message.reply_text(f'Точность модели: {accuracy:.2f}')

def delete_data(update: Update, _: CallbackContext) -> None:
    global df
    df = pd.DataFrame(columns=['text', 'label'])
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    update.message.reply_text('Все данные и модель удалены.')

def show_log(update: Update, _: CallbackContext) -> None:
    if os.path.exists('predictions.log'):
        with open('predictions.log', 'r') as log_file:
            logs = log_file.read()
            update.message.reply_text(f'Журнал предсказаний:\n{logs}')
    else:
        update.message.reply_text('Журнал предсказаний пуст.')

def love_message(context: CallbackContext) -> None:
    chat_id = '@user1names1'
    context.bot.send_message(chat_id=chat_id, text='я тебя люблю')

def tarot(update: Update, _: CallbackContext) -> None:
    card = random.choice(list(tarot_cards.keys()))
    meaning = tarot_cards[card]
    update.message.reply_text(f'Ваша карта Таро: {card}\n\nЗначение: {meaning}')

def main() -> None:
    updater = Updater("6621926449:AAHUaZWBXUFpt1IMQVShAkxrmFQb1O60H60telegram_bot.pytelegram_bot.py")

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("train", train))
    dispatcher.add_handler(CommandHandler("labels", show_labels))
    dispatcher.add_handler(CommandHandler("accuracy", show_accuracy))
    dispatcher.add_handler(CommandHandler("delete_data", delete_data))
    dispatcher.add_handler(CommandHandler("log", show_log))
    dispatcher.add_handler(CommandHandler("tarot", tarot))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, predict))

    job_queue = updater.job_queue
    job_queue.run_repeating(love_message, interval=3600, first=0)

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
