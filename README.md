# ML-Final-Project

# Predicting Personality from Text: MBTI vs. Big Five

This project investigates to what extent personality traits can be inferred from short-form online text using machine learning, and how predictive performance differs between categorical personality frameworks (MBTI) and continuous trait frameworks (Big Five / OCEAN). Using Reddit posts as naturalistic text data, we fine-tune ***BERT-based models*** to predict personality dimensions directly from language.

Our results show that personality prediction consistently exceeds chance, confirming that online language contains weak but detectable personality signals. Performance is stronger for MBTI, where stable categorical boundaries simplify prediction, while Big Five regression proves more challenging due to its continuous nature and the noise inherent in short, informal posts. Overall, pretrained transformers reliably extract psychologically meaningful cues from social media text.

---

 ## Our Output
 ```
  Training model for E_I...
  Some weights of BertForSequenceClassification were not initialized from the model checkpoint at prajjwal1/bert-tiny and are newly initialized: ['classifier.bias', 'classifier.weight']
  You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
  Epoch 1 done, Loss: 0.6925
  Epoch 2 done, Loss: 0.7044
  Epoch 3 done, Loss: 0.7487
  
  Evaluating model...
  Accuracy: 0.5986
  PR AUC:  0.1944
  
  Training model for S_N...
  Some weights of BertForSequenceClassification were not initialized from the model checkpoint at prajjwal1/bert-tiny and are newly initialized: ['classifier.bias', 'classifier.weight']
  You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
  Epoch 1 done, Loss: 0.5469
  Epoch 2 done, Loss: 0.3736
  Epoch 3 done, Loss: 0.5740
  
  Evaluating model...
  Accuracy: 0.5764
  PR AUC:  0.9330
  
  Training model for T_F...
  Some weights of BertForSequenceClassification were not initialized from the model checkpoint at prajjwal1/bert-tiny and are newly initialized: ['classifier.bias', 'classifier.weight']
  You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
  Epoch 1 done, Loss: 0.6766
  Epoch 2 done, Loss: 0.6377
  Epoch 3 done, Loss: 0.6287
  
  Evaluating model...
  Accuracy: 0.5529
  PR AUC:  0.5233
  
  Training model for J_P...
  Some weights of BertForSequenceClassification were not initialized from the model checkpoint at prajjwal1/bert-tiny and are newly initialized: ['classifier.bias', 'classifier.weight']
  You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
  Epoch 1 done, Loss: 0.6997
  Epoch 2 done, Loss: 0.7109
  Epoch 3 done, Loss: 0.6770
  
  Evaluating model...
  Accuracy: 0.5429
  PR AUC:  0.6491
  
  Post: Ok ya not sure if this model is actually predicting anything
  Predicted MBTI: ISTP
  
  Post: I love leading teams and planning long-term goals.
  Predicted MBTI: ENTP
  
  Post: I really enjoy spending time alone thinking about ideas.
  Predicted MBTI: ISTJ
  
  Post: whatever this is sooo annoying
  Predicted MBTI: INTP
  
  Post: Wait Im actually so excited for thiss
  Predicted MBTI: ISFP
 ```

 ## Our Evaluation Graphs

 **MBTI:**
 
<img src="pr_curve_E_I.png" alt="E/I PR Curve" width="600"/>

**Figure 1:** E/I Precision-Recall Curve

<img src="pr_curve_S_N.png" alt="S/N PR Curve" width="600"/>

**Figure 2:** S/N Precision-Recall Curve

<img src="pr_curve_T_F.png" alt="T/F PR Curve" width="600"/>

**Figure 3:** T/F Precision-Recall Curve

<img src="pr_curve_J_P.png" alt="J/P PR Curve" width="600"/>

**Figure 4:** J/P Precision-Recall Curve

**Big Five:**

**ADD THE GRAPHS OUR FINAL MODEL WILL OUTPUT**

## Replication Instructions

**1. Clone this repository:**
   ```
   git clone https://github.com/paigemizutani/ML-Final-Project.git
   ```
   ```
   cd ML-Final-Project
   ```
**2. Install the required libraries:**
   ```
   pip install torch transformers datasets pandas numpy scikit-learn matplotlib seaborn scipy tqdm
   ```
**3. Prepare the Datasets**

   **MBTI**
   - The balanced MBTI datasets used in our experiments is already included in the repository as preprocessed CSV files under `data/balanced`.
   - These files were generated during development and are provided to allow direct replication of results without rerunning `data_setup.py`.
   - To reproduce our results, no additional MBTI data setup is required.

   **Big Five**
   - For the Big Five model, dataset preparation is handled automatically within the training script.
   - The code downloads and processes the Pandora-Big5 Reddit dataset from Hugging Face and creates the required train/validation/test splits at runtime.
   - No manual data preparation is required.

**4. For Generating Predictions on New Text**

Both models support inference on new, raw text inputs. To input your own text edit the `new_posts` lists in `mbti_model.py` and `big5_model.py`.

**5. Run Training and Evaluation**

  Train and evaluate the MBTI model:
  ```
  python mbti_model.py
  ```
  Train and evaluate the Big Five model:
  ```
  python big5_model.py
  ```
  Each script trains a BERT-based model, evaluates performance on the test set, and outputs relevant metrics and visualizations.
 
  Note: We trained our models using Colgate University's super computer, and even with GPU acceleration, full training runs takes a couple hours.

 
 ## Future Directions

Future work could improve generalization by training on more diverse text sources beyond Reddit, such as blogs, essays, or social media platforms with longer-form content. Additionally, cross-framework analysis could explore correlations between MBTI dimensions and Big Five traits to identify shared linguistic signals and better understand how categorical and continuous personality models relate. Finally, incorporating demographic metadata (e.g., age groups or online communities) could help determine whether personality cues are expressed differently across populations.

---

## Contributions

This project was completed collaboratively, with both group members working together on all stages of the project, including dataset selection, model development, training, evaluation, analysis, and poster preparation. Work sessions were conducted jointly, and responsibilities were shared throughout.

The only division of work occurred during visualization: Paige created the plots and figures for the MBTI model results, while Ashley created the plots and figures for the Big Five model results.
- **Total estimated time**: 40 hours





