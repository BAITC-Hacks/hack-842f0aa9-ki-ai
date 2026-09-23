"""Local evidence chat; source files are read anew for each submitted question."""
from pathlib import Path

import streamlit as st

from src.assistant import SOURCES, answer_question


def assistant_panel(out_dir: Path, selected_gid: int | None) -> None:
    st.subheader('Дерек көмекшісі')
    st.caption('API-сіз анықтамалық режим. Бір gid бойынша есептелген себептерді, шектеулерді және тұрақтылықты көрсетеді. Генеративті ЖИ қолданылмайды.')
    missing = [name for name in SOURCES if not (out_dir / name).is_file()]
    if missing:
        st.info('Көмекшіні қосу үшін пайплайнды іске қосыңыз. Жетіспейтін файлдар: ' + ', '.join(missing))
        st.code('python pipeline.py --data data --out out')
        return
    snapshot = tuple((str((out_dir / name).resolve()), (out_dir / name).stat().st_mtime_ns,
                      (out_dir / name).stat().st_size) for name in SOURCES)
    if st.session_state.get('assistant_snapshot') != snapshot:
        st.session_state['assistant_snapshot'] = snapshot
        st.session_state['assistant_messages'] = []
    history = st.session_state.setdefault('assistant_messages', [])
    if selected_gid is not None:
        st.caption(f'Таңдалған клиент: {selected_gid}. Мысал: «{selected_gid} неге тексеру керек?»')
    if st.button('Сөйлесуді тазарту', key='clear_assistant'):
        history.clear()
    question = st.chat_input('Бір клиенттің толық gid мәнін жазыңыз', key='assistant_question', max_chars=1000)
    if question:
        history.append({'question': question, 'response': answer_question(question, out_dir)})
        del history[:-10]
    for entry in history:
        with st.chat_message('user'):
            st.text(entry['question'])
        with st.chat_message('assistant'):
            response = entry['response']
            st.text(response['answer'])
            if response['sources']:
                st.caption('Дереккөздер: ' + ' · '.join(
                    f"{source['file']} (gid={source['gid']})" for source in response['sources']))
