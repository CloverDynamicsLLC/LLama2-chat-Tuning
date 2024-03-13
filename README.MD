![Clover Logo](https://d64hhk2r5btyz.cloudfront.net/logo_f4de25a3df.svg)

# Llama-2-7b-chat Fine-Tuning Project

## Introduction
This project is an innovative approach to enhancing the Llama-2-7b-chat model, leveraging the Cornell Movie-Dialogs Corpus. The primary objective is to refine the model's conversational abilities, making it not only responsive but also proactive in engaging with users. By training the model to ask questions and sustain the conversation, we aim to create a more dynamic and interactive chatbot experience.

## Features
- **Dataset Preparation**: Detailed Python scripts for cleaning and structuring the Cornell Movie-Dialogs Corpus, ensuring optimal input data quality for model training.
- **Model Fine-Tuning**: A comprehensive Jupyter Notebook designed for the fine-tuning of the Llama-2-7b-chat model using QLoRA, specifically configured for execution on a Google Vertex AI notebook instance with 'n1-standard-4' machine type and an NVIDIA T4 GPU.
- **Monitoring and Analysis**: Integration with WandB for real-time monitoring and analysis of the training process, providing insights into model performance and metrics.
- **Customizable Interaction**: Post-training, the model is equipped to conduct conversations in a more friendly and engaging manner, with capabilities to both respond and initiate dialogue.

## Prerequisites
- **Access to Llama-2 Model**: Users must have access to the Llama-2 model on Hugging Face. Ensure you have the necessary permissions to use this model.
- **Access Tokens**: Generate and securely store access tokens for both Hugging Face and WandB, as these are crucial for model access and training monitoring.
- **Google Vertex AI Notebook Instance**: Set up a Google Vertex AI notebook instance as specified.

## Usage
1. **Dataset Preparation**: Execute the provided Python scripts to prepare the Cornell Movie-Dialogs Corpus.
2. **Model Fine-Tuning**:
    - Launch the Jupyter Notebook in the Google Vertex AI environment.
    - Upload the rest of the necessary files to the notebook instance.
3. **Training Monitoring**: Use WandB to monitor the training process and analyze the performance metrics.

## Resources
- [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- [QLoRA: Quantized Linear Transformers for Model Parallel Training](https://arxiv.org/abs/2110.14480)
- [Google Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Hugging Face - Llama-2-7b](https://huggingface.co/llama-2-7b)
- [WandB Documentation](https://docs.wandb.ai/)

## Support
For support and queries, please open an issue in the GitHub repository.

## Contributing
Contributions are welcome. Please fork the repository and submit a pull request with your changes.

## License
[MIT License](https://opensource.org/licenses/MIT)

---

*This project is not affiliated with Hugging Face, Google Cloud, Cornell University, or WandB. All product names, logos, and brands are property of their respective owners.*
