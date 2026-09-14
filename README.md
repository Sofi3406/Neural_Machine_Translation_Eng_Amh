# English to Amharic Neural Machine Translation

An attention-based sequence-to-sequence LSTM model that translates short English sentences into Amharic. The project includes a Streamlit interface, the trained model artifacts, and the original training notebook.

## Demo

The app is designed for deployment on [Streamlit Community Cloud](https://streamlit.io/cloud).

## Project structure

```text
.
├── app.py                              # Streamlit application
├── inference.py                        # Model definition and inference
├── requirements.txt                    # Deployment dependencies
├── deployment_artifacts/
│   ├── attention_model.pt              # Trained model weights
│   ├── config.pkl                      # Model configuration
│   ├── src_vocab.pkl                   # English vocabulary
│   └── trg_vocab.pkl                   # Amharic vocabulary
├── NMT_English_to_Amharic.ipynb        # Training and evaluation notebook
└── README.md
```

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows Git Bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io).
3. Select the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.

The model files in `deployment_artifacts/` are required at runtime. Keep them in the repository, or move them to a private model-storage service and update `app.py` and `inference.py` accordingly.

## Model notes

The notebook documents preprocessing, vocabulary creation, training, attention-based decoding, and evaluation. The Streamlit app loads the model once with `st.cache_resource`, then performs CPU inference when no GPU is available.

## License

Add the license that applies to the training data, model, and source code before publishing this repository.