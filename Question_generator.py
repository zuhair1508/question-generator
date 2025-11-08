from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os


app = FastAPI()

class AssessmentQuestionGenerator:
    def __init__(self):
        data_analyst_mcq = [
    {
        "question": "Create five questions for a Data Analyst assessment test",
        "answer": """
Question 1. What is the primary goal of exploratory data analysis (EDA)?
a) To build predictive models
b) To summarize the main characteristics of the dataset
c) To assess the statistical significance of relationships in the data
d) To clean and preprocess the data

Question 2. What is the primary goal of data cleaning in the data analysis process?
a) To remove outliers from the dataset
b) To organize data into a structured format
c) To identify and correct errors in the data
d) To visualize data patterns

Question 3. What is the purpose of data imputation in data analysis?
a) To remove outliers from the dataset
b) To replace missing values with estimated values
c) To standardize the scale of the data
d) To identify patterns in the data

Question 4. Which data visualization technique is most appropriate for comparing the distribution of a categorical variable across different groups?
a) Histogram
b) Bar chart
c) Scatter plot
d) Box plot

Question 5. What is the main benefit of using SQL (Structured Query Language) for data analysis?
a) It allows for complex statistical analysis
b) It facilitates data visualization
c) It enables efficient querying and manipulation of databases
d) It automates data cleaning processes

""",
    },
    {
        "question": "Create five questions for a Data Analyst assessment test",
        "answer": """
Question 1. What is the purpose of data imputation in data analysis?
a) To remove outliers from the dataset
b) To replace missing values with estimated values
c) To standardize the scale of the data
d) To identify patterns in the data

Question 2. Which data preprocessing technique is used to transform skewed data distributions into approximately normal distributions?
a) Feature scaling
b) Principal component analysis (PCA)
c) One-hot encoding
d) Log transformation

Question 3. In hypothesis testing, what does the p-value represent?
a) The probability of observing the data given that the null hypothesis is true
b) The probability of rejecting the null hypothesis when it is true
c) The probability of accepting the null hypothesis when it is false
d) The probability of obtaining a sample mean greater than or equal to the observed mean

Question 4. Which statistical measure is used to assess the strength and direction of the linear relationship between two continuous variables?
a) Pearson correlation coefficient
b) Chi-square statistic
c) F-statistic
d) Z-score

Question 5. In data analysis, what does the term "overfitting" refer to?
a) When a model is too simple to capture the underlying patterns in the data
b) When a model performs well on the training data but poorly on new data
c) When a dataset contains too many missing values
d) When there is a strong linear relationship between two variables

""",
    },
]

        data_analyst_tof = [
    {
        "question": "Create five True or False questions for a Data Analyst assessment test",
        "answer": """
Q1. True or False: Correlation implies causation.
Q2. True or False: A p-value below 0.05 indicates strong evidence against the null hypothesis.
Q3. True or False: Data mining is the process of extracting meaningful patterns from large datasets.
Q4. True or False: In a normal distribution, the mean, median, and mode are equal.
Q5. True or False: Data aggregation involves combining multiple datasets into a single dataset.

""",
    },
]
        data_analyst_open = [
    {
        "question": "Create 2 Open-Ended questions for a Data Analyst assessment test",
        "answer": """
Question 1. You've been given access to a dataset containing information about customer transactions for an e-commerce website. The dataset includes variables such as customer demographics, product categories, purchase amounts, and timestamps. Your task is to analyze this dataset to identify trends, patterns, and insights that could help improve the website's performance and increase sales. Describe your approach to analyzing this dataset, including any data cleaning, preprocessing, and exploratory data analysis steps you would take. Additionally, discuss the potential business implications of your findings and how you would communicate them to stakeholders.
Question 2. Consider a scenario where you're working for a marketing firm that wants to launch a new advertising campaign targeting a specific demographic segment. You're tasked with analyzing market research data to identify the characteristics and preferences of this target audience. Describe your strategy for conducting this analysis, including the data sources you would use, the analytical techniques you would apply, and how you would ensure the accuracy and reliability of your findings. Additionally, discuss how you would present your insights to the marketing team to inform their campaign strategy decisions.
""",
    },
    {
        "question": "Create two Open-Ended questions for a Data Analyst assessment test",
        "answer": """
Question 1. You've been provided with a dataset containing sales data for a retail company over the past year. This dataset includes information such as customer demographics, purchase history, and sales channels. Your task is to analyze this dataset comprehensively and provide insights that could inform business decisions. Additionally, describe any data cleaning and preprocessing steps you would take before conducting the analysis. Finally, discuss potential challenges you might encounter during the analysis process and how you would address them.
Question 2. Imagine you're working for a healthcare organization that wants to improve patient outcomes and reduce hospital readmission rates. You're tasked with analyzing patient data to identify factors that contribute to readmissions. How would you approach this analysis? Describe the steps you would take, the statistical methods you would use, and how you would interpret and communicate your findings to stakeholders. Additionally, discuss any ethical considerations you would need to take into account when handling sensitive patient data.
""",
    },
]
        product_manager_mcq = [
    {
        "question": "Create five questions for a Tech Product Manager copywriter assessment test",
        "answer": """
Question 1.Which authentication method is commonly used in e-wallet apps to ensure security and user convenience?
a) Two-factor authentication (2FA)
b) OAuth 2.0
c) Single Sign-On (SSO)
d) Secure Socket Layer (SSL)

Question 2.Which mobile app development framework allows for building cross-platform apps with a single codebase?
a) Xamarin
b) Flutter
c) Ionic
d) PhoneGap

Question 3.What is the role of a product manager in an Agile development environment?
a) Writing code and developing features
b) Managing the product backlog and prioritizing user stories
c) Conducting user testing and quality assurance
d) Providing technical support to customers

Question 4.What is the purpose of implementing push notifications in an e-wallet app?
a) To display targeted advertisements to users
b) To alert users about account activity and transaction updates
c) To collect user data for analytics purposes
d) To enable social sharing features

Question 5.Which database technology is suitable for storing transactional data in real-time in an e-wallet app?
a) NoSQL
b) Relational databases
c) In-memory databases
d) Graph databases

""",
    },
    {
        "question": "Create five questions for a Tech Product Manager assessment test",
        "answer": """

Question 1.Which technology is commonly used for implementing push notifications in mobile applications?
a) Firebase Cloud Messaging (FCM)
b) Simple Notification Service (SNS)
c) Twilio
d) Pusher

Question 2.What is the purpose of implementing OAuth 2.0 in an e-wallet app?
a) To encrypt sensitive user data
b) To authenticate users and authorize third-party access to user accounts
c) To enable peer-to-peer payment transactions
d) To track user interactions for analytics purposes

Question 3.Which mobile development framework allows for building high-performance, native-like experiences using web technologies like HTML, CSS, and JavaScript?
a) Ionic Framework
b) Xamarin
c) Flutter
d) Apache Cordova

Question 4.What is the role of a JSON Web Token (JWT) in user authentication?
a) To store user credentials securely on the client-side
b) To track user sessions across multiple devices
c) To generate access tokens for API requests
d) To provide identity verification for users

Question 5.Which database technology offers offline data synchronization and real-time collaboration features suitable for mobile applications?
a) MongoDB
b) Firebase Realtime Database
c) Cassandra
d) Couchbase

""",
    },

]        
        product_manager_tof = [
    {
        "question": "Create five True or False questions for a Tech Product Manager assessment test",
        "answer": """
Q1.True or False: A/B testing is a common technique used in product management to compare two versions of a feature and determine which one performs better.
Q2.True or False: React Native is a framework for building native iOS applications using JavaScript.
Q3.True or False: Progressive Web Apps (PWAs) offer native-like experiences and can be installed on a user's device like native apps.
Q4.True or False: Two-factor authentication (2FA) requires users to provide two forms of identification, such as a password and a fingerprint, to access their accounts.
Q5.True or False: Cross-site scripting (XSS) attacks can compromise the security of e-wallet apps by injecting malicious scripts into web pages.


""",
    },
    {
        "question": "Create five True or False questions for a Data Analyst assessment test",
        "answer": """

Q1.True or False: Blockchain technology is commonly used in e-wallet apps to enhance transaction security and transparency.
Q2.True or False: Product managers are responsible for writing code and implementing features in the product.
Q3.True or False: In-app chat support is an essential feature of e-wallet apps for providing customer assistance.
Q4.True or False: User acceptance testing (UAT) is conducted by the development team to ensure the product meets quality standards before release.
Q5.True or False: Continuous integration (CI) and continuous deployment (CD) practices help streamline the development and release process by automating testing and deployment.

""",
    },
]
        product_manager_open = [
    {
        "question": "Create 2 Open-Ended questions for a Product manager assessment test",
        "answer": """
Question 1. Your team is planning to introduce a new feature in the e-wallet app that allows users to set up recurring payments for utility bills and subscription services. Describe the technical architecture and backend infrastructure required to support recurring payments, including scheduling, billing cycles, and notifications.
Question 2. As part of the e-wallet app's expansion into emerging markets, you're tasked with optimizing the app for low-bandwidth and intermittent internet connectivity. Discuss the technical strategies and optimizations for improving app performance, data synchronization, and offline capabilities in resource-constrained environments.

""",
    },
]
        marketing_copywriter_mcq = [
    {
        "question": "Create five questions for a Marketing copywriter assessment test",
        "answer": """
Question 1. What is the primary goal of copywriting in digital marketing?
A. To create visually appealing graphics
B. To generate organic traffic
C. To create compelling and persuasive written content that resonates with the target audience
D. To conduct market research

Question 2. How does SEO copywriting differ from traditional copywriting, and why is it important in digital marketing?
A. SEO copywriting focuses on creating content for social media platforms, while traditional copywriting focuses on print media.
B. SEO copywriting aims to improve search engine visibility by incorporating relevant keywords, while traditional copywriting does not.
C. SEO copywriting focuses on creating long-form content, while traditional copywriting focuses on short-form content.
D. SEO copywriting focuses on visual elements, while traditional copywriting focuses on text-only content.

Question 3. What is the concept of tone of voice in copywriting, and why is it important in maintaining brand consistency?
A. The concept of tone of voice refers to the volume at which content is delivered, and it's important for attracting attention.
B. Tone of voice is irrelevant in copywriting and does not affect brand consistency.
C. Tone of voice refers to the style and personality of written content, and it's important for maintaining brand identity and consistency.
D. Tone of voice refers to the language used in copywriting, and it's only important for offline marketing materials.

Question 4. How do you approach crafting effective headlines and subject lines for digital marketing campaigns?
A. By randomly selecting words and phrases
B. By conducting market research on competitors' headlines
C. By testing different variations through A/B testing
D. By using as many keywords as possible

Question 5. What role does storytelling play in copywriting for digital marketing?
A. Storytelling helps create emotional connections with the audience and humanize brands.
B. Storytelling is irrelevant in digital marketing
C. Storytelling only works for long-form content
D. Storytelling is used to create visual content for digital marketing campaigns.
""",
    },
]
        marketing_tof = [
    {
        "question": "Create five True or False questions for a Marketing Copywriter assessment test",
        "answer": """
Q1. True or False: The primary goal of copywriting in digital marketing is to create visually appealing graphics.
Q2. True or False: SEO copywriting focuses on improving search engine visibility by strategically incorporating relevant keywords.
Q3. True or False: Tone of voice in copywriting refers to the volume at which content is delivered.
Q4. True or False: A/B testing is a common approach used to craft effective headlines and subject lines for digital marketing campaigns.
Q5. True or False: Storytelling plays no significant role in copywriting for digital marketing.
""",
    },
]
        marketing_open = [
    {
        "question": "Create 2 Open-Ended questions for a Marketing copywriter assessment test",
        "answer": """
Question 1. Company X, an e-commerce retailer, is launching a new product line targeting environmentally conscious consumers. As the copywriter for this campaign, how would you approach crafting product descriptions and marketing copy to appeal to this audience while emphasizing the eco-friendly features of the products?
Question 2. Company Y, a tech startup, is launching a new mobile app aimed at fitness enthusiasts. As the copywriter for this project, how would you approach crafting app store descriptions, in-app messaging, and promotional content to attract and engage the target audience?
""",
    },
]
        self.roles = {
    "data_analyst": {
        "multiple_choice": data_analyst_mcq,
        "true_or_false": data_analyst_tof,
        "open_ended": data_analyst_open
    },
    "product_manager": {
        "multiple_choice": product_manager_mcq,
        "true_or_false": product_manager_tof,
        "open_ended": product_manager_open
    },
    "marketing_copywriter": {
        "multiple_choice": marketing_copywriter_mcq,
        "true_or_false": marketing_tof,
        "open_ended": marketing_open
    }
    }
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=api_key,temperature=0.7,)
        self.template = PromptTemplate(
        input_variables=["question", "answer"], 
        template= """
        You are an expert assessment designer. Create questions as part of a question bank to be used to assess candidates for a Harvard course on the subject. You must NEVER repeat the example questions. Always create questions based on the number of questions requested. You must always provide the perfect answers to the questions at the end. Please provide the "Questions" and "Answers" in JSON format. 
        Example: {question}
        {answer}
            
        You must provide the "Answers" in JSON format after listing down all the questions.
        """)
        self.JsonOutputParser= JsonOutputParser()
        

    def map_to_prompt(self, input_str, question_type):
        role_info = self.roles.get(input_str, None)
        if role_info:
            prompt = role_info.get(question_type, None)
            if prompt:
                return prompt
            else:
                return None  # Return None if prompt not found for the given question type
        else:
            return None  # Return None if role not recognized

    def create_few_shot_template(self, prompt):
        few_shot_template = FewShotPromptTemplate(
            examples=prompt,
            example_prompt=self.template,
            suffix="Question: {input}",
            input_variables=["input"],
        )
        return few_shot_template

    def generate_assessment_questions(self, few_shot_template_output, num_questions, role_info, question_type):
        chain = few_shot_template_output | self.llm | self.JsonOutputParser
        result = chain.invoke(f"Create {num_questions} {role_info} {question_type} assessment questions")
        return result

generator = AssessmentQuestionGenerator()

class AssessmentQuestionRequest(BaseModel):
    role: str
    question_type: str
    num_questions: int

@app.post("/generate_questions/")
async def generate_questions(request_data: AssessmentQuestionRequest):
    role = request_data.role
    question_type = request_data.question_type
    num_questions = request_data.num_questions

    prompt = generator.map_to_prompt(role, question_type)
    if prompt is None:
        raise HTTPException(status_code=400, detail="Role or question type not recognized")


    few_shot_template = generator.create_few_shot_template(prompt)
    questions = generator.generate_assessment_questions(few_shot_template, num_questions, role, question_type)
    
    return questions