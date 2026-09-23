"""Grounded AI chat with session-only credentials and a visible offline fallback."""
import os
from pathlib import Path

import streamlit as st

from src.assistant import SOURCES, answer_question, answer_with_ai
from src.llm import AIConfig, PROVIDERS


def _server_setting(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, ''))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return ''


def _forget_keys() -> None:
    for provider in PROVIDERS:
        st.session_state.pop('ai_key_' + provider, None)
    st.session_state['ai_enabled'] = False


def assistant_panel(out_dir: Path, selected_gid: int | None, data_dir: Path | None = None) -> None:
    st.subheader('ЖИ көмекшісі')
    st.caption('Клиентті түсіндіру, бірнеше gid-ті салыстыру және ортақ тікелей алушыларды талдау. Жауапқа есептелген деректер ғана беріледі.')
    with st.expander('ЖИ-ді қосу: API баптаулары', expanded=not st.session_state.get('ai_enabled', False)):
        provider = st.selectbox('ЖИ провайдері', list(PROVIDERS), key='ai_provider')
        server_key = _server_setting(PROVIDERS[provider][2])
        typed_key = st.text_input('API кілті', type='password', key='ai_key_' + provider,
                                 help='Кілт осы браузер сессиясының жадында ғана сақталады. GitHub-қа немесе чат тарихына жазылмайды.')
        api_key = typed_key.strip() or server_key
        if server_key:
            st.caption('Сервер баптауларындағы кілт қолжетімді.')
        model = st.text_input('Модель', value=PROVIDERS[provider][1], key='ai_model_' + provider)
        enabled = st.toggle('ЖИ режимін қосу', key='ai_enabled')
        st.caption(f'ЖИ режимінде сұрақ пен ең көбі 5 клиенттің есептелген карточкасы {provider} сервисіне жіберіледі. Бірнеше gid үшін ортақ тікелей алушылардың ең көбі 10 жолы қосылады. Толық Parquet файлдары жіберілмейді.')
        st.button('Енгізілген кілттерді өшіру', on_click=_forget_keys, key='forget_ai_keys')
    use_ai = enabled and bool(api_key)
    if enabled and not api_key:
        st.warning('ЖИ қосылуы үшін API кілтін енгізіңіз. Қазір жергілікті жауап беру режимі жұмыс істейді.')
    elif use_ai:
        st.info(f'ЖИ режимі: {provider} · {model}. Қосылу келесі сұрақ жіберілгенде тексеріледі.')
    else:
        st.caption('Қазір API-сіз режим: бір толық gid бойынша дайын карточка көрсетіледі. ЖИ үшін жоғарыдағы баптауларды ашыңыз.')
    missing = [name for name in SOURCES if not (out_dir / name).is_file()]
    if missing:
        st.info('Көмекшіні қосу үшін пайплайнды іске қосыңыз. Жетіспейтін файлдар: ' + ', '.join(missing))
        st.code('python pipeline.py --data data --out out')
        return
    context_files = [out_dir / name for name in SOURCES]
    if data_dir is not None and (data_dir / 'edges.parquet').is_file():
        context_files.append(data_dir / 'edges.parquet')
    snapshot = tuple((str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size) for path in context_files)
    if st.session_state.get('assistant_snapshot') != snapshot:
        st.session_state['assistant_snapshot'] = snapshot
        st.session_state['assistant_messages'] = []
    history = st.session_state.setdefault('assistant_messages', [])
    if selected_gid is not None:
        st.caption(f'Таңдалған клиент: {selected_gid}. Мысал: «{selected_gid} неге тексеру керек?»')
    st.caption('Мысалдар: «Неге осы клиентті тексеру керек?», «gid1 мен gid2-ні салыстыр», «Осы gid-тердің ортақ алушысы кім?». Клиент таңдалмаса, ЖИ басымдығы жоғары 5 клиентті ғана көреді.')
    if st.button('Сөйлесуді тазарту', key='clear_assistant'):
        history.clear()
    question = st.chat_input('Сұрақ жазыңыз немесе клиенттің gid мәнін енгізіңіз', key='assistant_question', max_chars=1000)
    if question:
        if use_ai:
            with st.spinner('ЖИ есептелген деректерді талдап жатыр…'):
                response = answer_with_ai(question, out_dir, config=AIConfig(provider, api_key, model),
                                          selected_gid=selected_gid, data_dir=data_dir)
        else:
            response = answer_question(question, out_dir)
        history.append({'question': question, 'response': response})
        del history[:-10]
    for entry in history:
        with st.chat_message('user'):
            st.text(entry['question'])
        with st.chat_message('assistant'):
            response = entry['response']
            if response.get('notice'):
                st.info(response['notice'])
            if response['mode'] == 'llm':
                st.caption(f"ЖИ жауабы · {response['provider']} · {response['model']}")
            else:
                st.caption('Жергілікті дерек жауабы · ЖИ генерациясы қолданылған жоқ')
            st.text(response['answer'])
            if response['sources']:
                st.caption('Дереккөздер: ' + ' · '.join(
                    f"{('[' + source['id'] + '] ') if 'id' in source else ''}{source['file']} (gid={source['gid']})"
                    for source in response['sources']))
            if response.get('evidence'):
                with st.expander('ЖИ қолданған нақты деректер'):
                    st.json(response['evidence'])
