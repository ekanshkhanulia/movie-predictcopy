
Hide Assignment Information
Group Category
Group
Group Name
Group 18
Instructions
Assignment 2: SASRec
🎯 Objective
In this assignment, you will implement a sequential recommendation model based on SASRec to predict the next item a user will interact with from their historical behavior sequence. SASRec is a strong self-attention-based sequential recommendation model that captures users’ dynamic preferences by modeling dependencies among previously interacted items.

Your goal is to:

Implement SASRec using PyTorch

Train and evaluate the model on the MovieLens 1M dataset

Evaluate the performance using NDCG@10, NDCG@20 and Recall@10, Recall@20

You may refer to the official SASRec implementation for guidance:
https://github.com/kang205/SASRec

📁 Dataset
Use the MovieLens 1M dataset, which can be downloaded from the MovieLens Official Website. The dataset format is:

userId::movieId::rating::timestamp
For this assignment:

Treat ratings ≥ 4 as positive interactions

Ignore all interactions with ratings < 4

Convert each user’s history into a chronologically ordered sequence of interacted items

🛠️ Implementation Requirements
1. Data Preprocessing (15 points)
Load and preprocess the MovieLens dataset

Convert explicit ratings to binary implicit feedback

Generate chronological interaction sequences for each user

Filter out users with fewer than 5 interactions

For each user sequence, apply a leave-one-out split:

Use all but the last two interactions for training

Use the second-to-last interaction for validation

Use the last interaction for testing

Construct input–target pairs for next-item prediction based on user sequences

2. SASRec Model (20 points)
Implement the SASRec architecture, including:

Item embeddings and positional embeddings

Self-attention blocks for modeling sequential dependencies

Causal attention masking so that each position only attends to previous items

Feedforward layers, dropout, and layer normalization

A final prediction layer that scores candidate items based on the sequence representation

Use PyTorch for implementation. You may not simply call a ready-made SASRec package without understanding and adapting the architecture yourself.

3. Training and Optimization (20 points)
Train the model using the next-item prediction objective

Use the Adam optimizer with appropriate learning rate tuning or scheduling

Implement early stopping based on validation NDCG@10

Use dropout and layer normalization appropriately

Use negative sampling during training

4. Evaluation (15 points)
Evaluate the model using:

Recall@10, Recall@20

NDCG@10, NDCG@20

For evaluation: (Full ranking)

Use the sequence prefix to predict the held-out next item in the validation and test sets

Compare performance across different configurations, such as:

number of self-attention blocks

hidden size

number of attention heads

maximum sequence length

5. Experiment Report (40 points)
Write a concise 2–3 page report summarizing your work (do not exceed 5 pages excluding references), including:

Preprocessing and sequence construction

Model architecture and implementation details

Training setup and hyperparameters

Evaluation results and comparison across settings

Insights, challenges, and suggestions for improvement

📝 Submission Requirements
Submit the following files:

Python source code (.py or .ipynb) with clear inline comments

Experiment report (.pdf) written using LaTeX, using the Springer LNCS Overleaf Template or ACM double-column format (https://www.overleaf.com/latex/templates/association-for-computing-machinery-acm-sig-proceedings-template/bmvfhcdnxfty).

A README.md file with instructions on how to run the code

📊 Grading Breakdown
Task	Points	Criteria
Data Preprocessing	15	Correctly generates sequences and dataset splits
SASRec Implementation	20	Accurately implements self-attention-based sequential recommendation
Training & Optimization	20	Trains model with appropriate objective and tuning strategies
Evaluation	15	Computes required metrics correctly and compares different settings
Experiment Report	40	Well-structured, concise, and insightful
Total	100	
⏰ Deadline & Submission
Submission Deadline: 27th April 2025, 23:59

Submit via Brightspace

Late Penalty: 5% deduction per day

Submissions made after 4th May 2025, 23:59 will receive 0 points

Due on 27 April 2026 23:59
