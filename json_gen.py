import json
import os
import shutil


def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)


data_to_save = \
    {
        "Year":
            """2025""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Fall""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Real Estate Price Prediction using Machine Learning""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """
            The goal of this project is to develop a Predictive model and a recommendation engine for real estate prices.
            The Predictive model will analyze historical real estate data to forecast future property prices based on various features such as location, size, number of rooms, and amenities.
            The recommendation engine will suggest optimal buying or selling times and strategies based on market trends and individual
            preferences. This project aims to provide valuable insights for buyers, sellers, and investors in the real estate market.
            """,
            
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            Companys Internal Dataset
            1. Property Listings: A comprehensive dataset containing historical and current property listings, including features such as location, size, number of rooms, amenities, and price.
            2. Market Trends: Data on real estate market trends, including average prices, demand-supply dynamics, and economic indicators.
            3. User Preferences: Data on user preferences and behaviors, including search history, saved properties, and transaction history.
            4. External Data Sources: Integration of external data sources such as census data, economic reports, and geographic information to enhance the predictive model.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            Semi-supervised learning is widely used when labeled data is limited, but manually labeling large datasets 
            is time-consuming and costly. Pseudo-labeling allows models to leverage unlabeled data by assigning estimated 
            labels, but existing heuristics may propagate errors and reduce model performance. 

            By applying reinforcement learning to pseudo-labeling, students can develop an adaptive strategy that selects 
            the most informative unlabeled examples and balances risk and reward in label assignment. This approach 
            reduces labeling errors, improves model accuracy, and provides a scalable solution for leveraging unlabeled 
            datasets in machine learning research.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            [Understanding the Reinforcement Learning (RL) Framework] 
            Students will learn how to formulate pseudo-labeling as an RL problem, including:

            - Understanding Markov Decision Process (MDP) assumptions and limitations: markov property, stationarity, limitations.
            - Understanding sequential decision making: actions affect future states and rewards.
            - Understanding terminal states and episodic tasks: pseudo-labeling epidode ends when all unlabeled data is processed.
            - State space design: selecting features of unlabeled examples, model predictions, and uncertainty measures 
            (e.g. [feature1, ..., featuren, softmax_predictions, cross_entropy]).
            - Action space design: assign a pseudo-label or skip labeling an example (e.g. {0, 1, ..., 9} for MNIST).
            - Reward design: reward high-confidence correct pseudo-labels and penalize incorrect assignments.
            - Algorithm selection: evaluate classical RL methods such as Q-learning, and deep RL methods such as Policy Gradient approaches.

            [`model.py` & `train.py`]
            Students will learn how to adapt existing RL repositories for pseudo-labeling:

            - Utilize existing RL repos (e.g. `twallett` GitHub Repo: `rl-lecture-code`) with existing RL models (e.g. Policy Gradient, PPO Clip).
            - Train RL agent on existing OpenAI Gymnasium environments (e.g. `CartPole-v1`) just to become familiar with code structure.

            [`utils/env.py` & `test.py`]
            Students will learn how to build a custom OpenAI Gymnasium environment for pseudo-labeling:

            - PseudoLabelEnv() & __init__() -> class: Initialize Custom OpenAI Gymnasium environment.
            OpenAI Gymnasium Core Methods:
            - self.reset() -> method/func: Reset environment to initial state.
            - self.step(action) -> method/func: Take action (assign pseudo-label or skip) and return next_state, reward, done, info.
            - (OPTIONAL) self.render() -> method/func: visualization of environment state.
            - self.close() -> method/func: Clean up resources.

            Additional Custom Pseudo-Labeling Methods:
                Basic MDP Custom Methods:
                - self.load_data() -> method/func: Load labeled and unlabeled datasets.
                - self.split_data() -> method/func: Split data into training and test sets.
                - self.get_state() -> method/func: Return feature representations for RL state.
                - self.calculate_reward() -> method/func: Compute reward based on correctness of pseudo-label and downstream model performance.

                Downstream Model Custom Methods:
                - DownstreamModel() & __init__() -> class: Initialize Downstream DL Model (MLP or CNN).
                - self.train_downstream_model() -> method/func: Train model on labeled + pseudo-labeled data.

                Evaluation Custom Methods:
                - evaluate_pseudo_labels() -> method/func: Compare pseudo-labels with ground truth for evaluation.
                - get_cum_rew() -> method/func: Compute cumulative reward over an episode.
                - get_classification_metric_gain() -> method/func: Measure improvement in downstream model using pseudo-labels.

            - `test.py` similar to `train.py` but for evaluating trained RL agent on unseen data.

            [`benchmark.py`]
            Students will learn how to systematically evaluate pseudo-labeling approaches:

            - Run experiments with different RL algorithms and hyperparameters.
            - Run experiments with different datasets.
            - Record pseudo-labeling performance and downstream model classification metrics.
            - Export results for analysis.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            [Understanding  1 week
            [`model.py` & `train.py`] 2 weeks
            [`utils/env.py` & `test.py`] 7 weeks
            [`benchmark.py`] 2 weeks (start writting research paper here)
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            Goal is to work solo. 
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Research Contributions":
            """
            This project will contribute to machine learning research by providing an open-source RL framework for 
            adaptive pseudo-labeling in semi-supervised learning. The methodology will enable researchers to assign 
            high-quality pseudo-labels to unlabeled data, improving model performance and reducing labeling costs. 
            Research findings can be published in academic journals or conferences, and the framework will be made 
            available for future researchers to extend and apply to new datasets.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            - Reinforcement Learning concepts can be difficult for students to grasp, especially MDP assumptions.  
            - Designing state, action, and reward spaces for pseudo-labeling may be non-trivial.  
            - Training RL agents can be computationally expensive and time-consuming.  
            - Risk of overfitting to small datasets or poor generalization to new unlabeled data.  
            - Debugging custom environments and reward functions can be challenging.  
            - Evaluation of pseudo-label quality requires careful metric design.  
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Ujjawal Dwivedi",
        "Proposed by email": "ujjawal.dwivedi@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gwu.edu",
        "collaborator": "None",
        "funding_opportunity": "Internship Extension",
        "github_repo": "fall-2025-group3",
        # -----------------------------------------------------------------------------------------------------------------------
    }
os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy('json_gen.py', output_file_path)
print(f"Data saved to {output_file_path}")