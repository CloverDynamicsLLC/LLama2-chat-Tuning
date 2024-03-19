![Clover Logo](https://d64hhk2r5btyz.cloudfront.net/logo_f4de25a3df.svg)

# Llama-2-7b-chat Fine-Tuning Project

## Introduction
Welcome to our project, where we embark on an exciting journey to enhance the conversational abilities of the Llama 2 model. Leveraging the richness of the Cornell Movie Dialog Corpus, our aim is to transcend the traditional boundaries of AI responses, transforming Llama 2 into not just a question-answering machine, but a friendly, engaging companion.

## Motivation
In its base form, the Llama 2 model is proficient in providing informative, interview-style answers. However, it often lacks the depth and continuity needed for more dynamic, human-like conversations. Recognizing this gap, we set out to fine-tune the model to foster an experience that's not just informative, but also immersive and relational.

## Challenges
1. **Adapting to Conversational Context**: One significant hurdle is teaching Llama 2 to understand and respond to the ebb and flow of casual dialogue, rather than just answering questions.
2. **Maintaining Coherency Over Long Conversations**: Ensuring that the model can hold and reference past parts of the conversation for a more connected and coherent dialogue experience.
3. **Injecting Personality and Engagement**: Infusing the model with a more relatable, friendly tone, departing from the often mechanical nature of AI responses.

## Solution: Fine-Tuning with the Cornell Movie Dialog Corpus
The Cornell Movie Dialog Corpus, with its vast array of conversational exchanges from movie scripts, presents an ideal dataset for this undertaking. By fine-tuning Llama 2 on these dialogues, we aim to:
- Enhance the model's ability to participate in and sustain long-form conversations.
- Develop a more nuanced understanding of context and emotional cues in dialogues.
- Cultivate a conversational style that's more aligned with a friendly companion rather than a factual respondent.

## Project Outcomes and Goals
Our ultimate goal is to reshape the Llama 2 model into an AI that users find relatable, engaging, and capable of maintaining satisfying, lifelike conversations. We believe this advancement will not only elevate the user experience but also open new avenues in the realm of conversational AI.


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
