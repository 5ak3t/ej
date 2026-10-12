# ej

**CPU上で型付きの判断を行う11 MBモデル。LLMもトークンも不要です。**

<img src="https://raw.githubusercontent.com/5ak3t/ej/main/docs/ej-demo.gif" alt="ejが3件のサポートチケットを請求、返品、テクニカルサポートへ振り分け、各チームの確率を表示" width="800">

```bash
pip install ejai
```

```python
import ej

model = ej.load('5ak3t/ej', revision='v0.0.1')
q = {'type': 'choice', 'instructions': 'Which team should handle this ticket?',
     'options': [{'key': 'billing', 'text': 'Billing and payments'}, {'key': 'tech', 'text': 'Technical problems'}]}
(probs,) = model.predict([{'state': 'I was charged twice this month.', 'questions': {'team': q}}])
print(probs['team'])   # [P(billing), P(tech)]: about [0.97, 0.03]
```

<p>
  <a href="https://github.com/5ak3t/ej/actions/workflows/tests.yml"><img alt="tests" src="https://img.shields.io/github/actions/workflow/status/5ak3t/ej/tests.yml?style=for-the-badge&labelColor=000000&label=tests" height="28"></a>
  <a href="https://huggingface.co/5ak3t/ej/tree/v0.0.1"><img alt="Weights: Hugging Face 5ak3t/ej v0.0.1" src="https://img.shields.io/badge/WEIGHTS-5ak3t%2Fej%20v0.0.1-0a0a0a.svg?style=for-the-badge&labelColor=000000&logo=huggingface" height="28"></a>
  <a href="../../LICENSE"><img alt="Code: GPL-3.0" src="https://img.shields.io/badge/code-GPL--3.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="../../LICENSES/CC-BY-SA-4.0.txt"><img alt="Weights: CC BY-SA 4.0" src="https://img.shields.io/badge/weights-CC%20BY--SA%204.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

ej に `state`（自由文、またはテキストとしての JSON オブジェクト）と型付きの質問を渡すと、質問ごとに1つの確率分布を返します。推論時に LLM は使いません。このリポジトリにはパッケージ、学習・評価コード、ベンチマークが含まれます。重みは Hugging Face Hub にあります。

アーキテクチャと手法: 技術レポートを準備中です。

ej は英語のみをサポートします。

## Quickstart

Python >= 3.10（3.11.15、CPUでテスト済み）。`pyproject.toml` の下限はテスト済みバージョンです。

```bash
pip install ejai            # the PyPI name is ejai; the import name is ej.  'ejai[eval]', 'ejai[train]' add the extras
# from source:
git clone https://github.com/5ak3t/ej && cd ej
pip install -e .            # inference;  -e '.[eval]' adds the evaluator, -e '.[train]' the training extras
```

`ej.load('5ak3t/ej', revision='v0.0.1')` は公開済みの `model.ejpack`（そのファイルだけを Hugging Face のキャッシュに保存）をダウンロードし、SHA-256を検証します。ローカルの `model.ejpack`、または自分で作成したファイル（`python -m ej.train fit`、続けて `python -m ej.train export`。後述）も、パスから同じ方法で読み込めます。

```python
import ej

model = ej.load('5ak3t/ej', revision='v0.0.1')   # or a local path: ej.load('model.ejpack')
record = {
    'state': '{"customer_tier": "gold", "message": "The blender arrived with a cracked jug. Replace it before Friday."}',
    'questions': {
        'route': {'type': 'choice', 'instructions': 'Which team should handle this request?',
                  'options': [{'key': 'returns', 'text': 'Returns and replacements for damaged or wrong items'},
                              {'key': 'billing', 'text': 'Billing, invoices and payment problems'}]},
        'needs_human': {'type': 'noul', 'instructions': 'The customer is upset enough that a human agent should reply.',
                        'options': ej.NOUL_OPTIONS},
        'urgency': {'type': 'score', 'instructions': 'How urgent is this request?',
                    'options': [{'key': '0', 'text': 'Low: can wait a week'},
                                {'key': '1', 'text': 'Medium: answer within two days'},
                                {'key': '2', 'text': 'High: answer today'}]},
    },
}
(probs,) = model.predict([record])
print(probs['route'])          # [p_returns, p_billing], sums to 1
```

`python examples/quickstart.py --weights 5ak3t/ej --revision v0.0.1 [--records file.jsonl]` は `ej.EXAMPLE_RECORD` または指定したファイル（ローカルファイルには `--weights model.ejpack`）を使って同じ処理を行います。`ej.load('model.ejpack', low_memory=True)` は同じ予測をより少ないメモリで行いますが、時間がかかります。重みのディレクトリ（safetensors + JSON、`fit` が書き出す形式）も読み込めます。この形式ではベースモデルからエンコーダーを構築するため、初回使用時に Hugging Face Hub から取得します。

**入力。** `{"id": "optional", "state": "...", "questions": {"<qid>": {"type": "choice | noul | score", "instructions": "...", "options": [{"key": "k1", "text": "short description"}, ...]}}}`。各質問には少なくとも2つの選択肢が必要で、キーは一意でなければなりません。`noul` の選択肢は `false` と `true`（`ej.NOUL_OPTIONS`）だけです。`score` の選択肢は低いものから順に並べます。レコードは `ej.validate_records` によって検証されます（エラー時は `ej.RecordError`）。**出力。** レコードごとに1つの辞書 `{qid: [p_1, ..., p_K]}` を返します。選択肢の順序に対応した通常の浮動小数点数で、合計は1（誤差1e-9以内）です。

**決定性とプロセス管理。** スレッド数とチャンク分割を固定すると予測は決定的です。チャンクサイズ、バッチ構成、スレッド数による確率の変化は浮動小数点のノイズ程度です（測定した最大 |Δp| は約6e-7）。`ej.load` は torch のスレッド数、乱数状態、`transformers` を変更しません。`predict(records, threads=None, chunk_size=None)` は呼び出し中だけスレッド数を設定します。1プロセスにつきロードするモデルは1つです。環境変数: `EJ_CACHE`（既定値 `~/.cache/ej`、パックの tokenizer ファイルを保存）、`EJ_COLD=1`（予測時キャッシュをバイパス）、`HF_HOME`（ダウンロードした `model.ejpack` と、重みのディレクトリで使うベースモデルの Hugging Face キャッシュ）、`HF_HUB_OFFLINE=1`（キャッシュがある場合にオフラインで実行）。
