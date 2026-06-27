# src/got/prompter.py

"""
Шаблоны промптов для операций Graph of Thoughts.
"""
class Prompter:
    def format_generate(self, context: str, question: str) -> str:
        return f"Используя следующий контекст:\n{context}\n\nОтветь на вопрос: {question}"

    def format_generate_with_role(
        self,
        base_prompt: str,
        question: str,
        role_name: str,
        role_description: str
    ) -> str:
        return f"""
{base_prompt}

Ты выступаешь в роли **{role_name}**.
{role_description}

Дай свой ответ на вопрос, исходя из этой роли. Ответ должен быть развёрнутым, но чётким.
Вопрос: {question}
Ответ:
"""

    def format_score(self, thought: str, question: str) -> str:
        return f"""
Оцени следующий ответ на вопрос '{question}' по шкале от 0 до 10.
0 – абсолютно неверно, 10 – идеально, полно, логично и без ошибок.
Верни только число (например, 8.5) без пояснений.
Ответ: {thought}
"""

    def format_refine(self, thought: str, question: str) -> str:
        return f"""
Улучши следующий ответ на вопрос '{question}'.
Сделай его более точным, полным, логичным и хорошо структурированным.
Исправь возможные ошибки и добавь важные детали, если их не хватает.
Верни только улучшенный ответ, без лишних пояснений.
Исходный ответ: {thought}
Улучшенный ответ:
"""

    def format_merge(self, thoughts: list, question: str) -> str:
        thoughts_text = "\n".join([f"- {t}" for t in thoughts])
        return f"""
Ниже представлены несколько вариантов ответа на вопрос '{question}'.
Объедини их в один связный, полный и лучший ответ, который вбирает в себя сильные стороны каждого варианта.
Избегай повторений, выбери наилучшие формулировки.
Верни только итоговый ответ.
Варианты:
{thoughts_text}
Итоговый ответ:
"""