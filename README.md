# Answer Evaluation System 📚✨

Welcome to the **Answer Evaluation System** repository! This project leverages advanced **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques to accurately evaluate student answers. Our system achieves an impressive accuracy rate of 89%. Below, you'll find detailed information on the project, including setup instructions, an overview of ML and NLP, and the technologies used.

![Project Screenshot](screenshot/preview.gif)

## Table of Contents

- [What is Machine Learning?](#what-is-machine-learning)
- [What is Natural Language Processing?](#what-is-natural-language-processing)
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Contributing](#contributing)
- [License](#license)

## What is Machine Learning? 🤖

**Machine Learning (ML)** is a subset of artificial intelligence (AI) that focuses on building systems that can learn from and make decisions based on data. ML algorithms use statistical techniques to identify patterns and make predictions or decisions without being explicitly programmed to perform the task.

## What is Natural Language Processing? 🗣️

**Natural Language Processing (NLP)** is a branch of artificial intelligence that helps computers understand, interpret, and respond to human language. NLP combines computational linguistics, computer science, and statistical modeling to process and analyze large amounts of natural language data.

## Project Overview 🌟

The **Answer Evaluation System** uses a variety of NLP techniques to evaluate student responses against expected answers. The system calculates multiple scores based on different criteria and then combines these scores using a weighted average to provide a final evaluation score. The key features include:

- **Preprocessing Text**: Tokenization and Lemmatization
- **Exact and Partial Match**: Comparing the student’s answer to the expected answer
- **Cosine Similarity**: Measuring similarity between texts
- **Sentiment Analysis**: Evaluating the sentiment of the response
- **Enhanced Sentence Match**: Using pre-trained models for semantic similarity
- **Multinomial Naive Bayes**: Probabilistic analysis
- **Coherence and Relevance Scores**: Assessing logical flow and content relevance

## Technologies Used 🚀

- **Flask**: Micro web framework for Python
- **Python**: Primary programming language
- **Jupyter Notebook (ipynb)**: Interactive computational environment
- **HTML, CSS, Bootstrap**: Frontend development
- **Gemini AI**: For advanced NLP models
- **Machine Learning & NLP**: Core of the evaluation system
- **SQL**: Database management

## Installation 💻

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/answer-evaluation-system.git
    cd answer-evaluation-system
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```


4. **Run the application**:
    ```bash
    python admin.py
    ```

# 🧠 Teacher Part Application - Hybrid NLP Evaluation System

This project is a **Hybrid NLP Framework** designed for automated evaluation of subjective answers.  
It uses a combination of **lexical** and **semantic similarity techniques** to compare a student’s answer with the expected answer.

---

## ⚙️ Step 1: Download and Install XAMPP

1. Go to the official XAMPP website:  
   👉 [https://www.apachefriends.org/index.html](https://www.apachefriends.org/index.html)
2. Download and install XAMPP for your operating system.
3. After installation, open the **XAMPP Control Panel**.

---

## 🚀 Step 2: Start Services

1. In the **XAMPP Control Panel**, find **Apache** and click **Start** (it should turn green).  
2. Then, find **MySQL** and click **Start** (it should also turn green).

---

## 🌐 Step 3: Open phpMyAdmin

1. Open your web browser.  
2. Type the following URL in the address bar and press Enter:  
http://localhost/phpmyadmin

sql
Copy code

---

## 🗃️ Step 4: Create the Database

1. In phpMyAdmin, look for the **"Create database"** section.  
2. Type `teacher_part` as the database name.  
3. Click **Create**.

---

## 🧩 Step 5: Create the Tables

1. After creating the database, click on **teacher_part** from the left sidebar.  
2. Go to the **SQL** tab.  
3. Copy and paste the following SQL code:

```sql
-- Create the Admins table
CREATE TABLE Admins (
    admin_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);

-- Create the Teachers table
CREATE TABLE Teachers (
    teacher_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);

-- Create the Students table
CREATE TABLE Students (
    student_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);

-- Create the Tests table
CREATE TABLE Tests (
    test_id INT AUTO_INCREMENT PRIMARY KEY,
    test_name VARCHAR(255) NOT NULL,
    teacher_id INT,
    FOREIGN KEY (teacher_id) REFERENCES Teachers(teacher_id) ON DELETE CASCADE
);

-- Create the Questions table
CREATE TABLE Questions (
    question_id INT AUTO_INCREMENT PRIMARY KEY,
    question_text TEXT NOT NULL,
    test_id INT,
    FOREIGN KEY (test_id) REFERENCES Tests(test_id) ON DELETE CASCADE
);

-- Create the ExpectedAnswers table
CREATE TABLE ExpectedAnswers (
    answer_id INT AUTO_INCREMENT PRIMARY KEY,
    answer_text TEXT NOT NULL,
    question_id INT,
    FOREIGN KEY (question_id) REFERENCES Questions(question_id) ON DELETE CASCADE
);

-- Create the StudentAnswers table
CREATE TABLE StudentAnswers (
    answer_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT,
    test_id INT,
    question_id INT,
    answer_text TEXT,
    score INT DEFAULT 0,
    FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (test_id) REFERENCES Tests(test_id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES Questions(question_id) ON DELETE CASCADE
);

-- Create the teacher-student relationship table
CREATE TABLE teacherstudentrelationship (
    relationship_id INT AUTO_INCREMENT PRIMARY KEY,
    teacher_id INT,
    student_id INT,
    FOREIGN KEY (teacher_id) REFERENCES Teachers(teacher_id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE
);
Click the Go button (bottom-right corner).
✅ This will create all required tables.

👨‍💼 Step 6: Manually Add Admin User
Make sure the teacher_part database is selected.

Click the SQL tab again.

Paste the following SQL command:

sql
Copy code
INSERT INTO Admins (username, password) VALUES ('admin', 'admin101');
Click Go.
✅ You have successfully added the default admin user.



## Usage 📚

1. Open your web browser and navigate to `http://127.0.0.1:5000`.
2. Enter the expected answer and the student’s answer in the provided fields.
3. Click "Evaluate" to get the evaluation score.

![Evaluation GIF](screenshot/evaluation_demo.gif)

## Code Overview 🧩

### Preprocess Text

```python
def preprocess_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return lemmatized_tokens
```

### Exact Match Function

```python
def exact_match(expected_answer, student_answer):
    return int(expected_answer == student_answer)
```

### Partial Match Function

```python
def partial_match(expected_answer, student_answer):
    expected_tokens = preprocess_text(expected_answer)
    student_tokens = preprocess_text(student_answer)
    common_tokens = set(expected_tokens) & set(student_tokens)
    match_percentage = len(common_tokens) / max(len(expected_tokens), len(student_tokens))
    return match_percentage
```

### Cosine Similarity Function

```python
def cosine_similarity_score(expected_answer, student_answer):
    vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
    tfidf_matrix = vectorizer.fit_transform([expected_answer, student_answer])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return cosine_sim
```

### Sentiment Analysis Function

```python
def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return (sentiment_score + 1) / 2
```

### Enhanced Sentence Match Function

```python
def enhanced_sentence_match(expected_answer, student_answer):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings_expected = model.encode([expected_answer])
    embeddings_student = model.encode([student_answer])
    similarity = cosine_similarity([embeddings_expected.flatten()], [embeddings_student.flatten()])[0][0]
    return similarity
```

### Multinomial Naive Bayes Score

```python
def multinomial_naive_bayes_score(expected_answer, student_answer):
    answers = [expected_answer, student_answer]
    vectorizer = CountVectorizer(tokenizer=preprocess_text)
    X = vectorizer.fit_transform(answers)
    y = [0, 1]
    clf = MultinomialNB()
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    return probs[1][1]
```

### Weighted Average Score Function

```python
def weighted_average_score(scores, weights):
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight
```

### Evaluation Function

```python
def evaluate(expected, response):
    if expected == response:
        return 10
    elif not response:
        return 0

    exact_match_score = exact_match(expected, response)
    partial_match_score = partial_match(expected, response)
    cosine_similarity_score_value = cosine_similarity_score(expected, response)
    sentiment_score = sentiment_analysis(response)
    enhanced_sentence_match_score = enhanced_sentence_match(expected, response)
    multinomial_naive_bayes_score_value = multinomial_naive_bayes_score(expected, response)
    semantic_similarity_value = semantic_similarity_score(expected, response)
    coherence_value = coherence_score(expected, response)
    relevance_value = relevance_score(expected, response)

    scores = [exact_match_score, partial_match_score, cosine_similarity_score_value, sentiment_score,
              enhanced_sentence_match_score, multinomial_naive_bayes_score_value, semantic_similarity_value,
              coherence_value, relevance_value]
    weights = [0.15, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1]

    scaled_scores = [score * 10 for score in scores]
    final_score = weighted_average_score(scaled_scores, weights)
    rounded_score = round(final_score)

    print("Exact Match Score:", exact_match_score)
    print("Partial Match Score:", partial_match_score)
    print("Cosine Similarity Score:", cosine_similarity_score_value)
    print("Sentiment Score:", sentiment_score)
    print("Enhanced Sentence Match Score:", enhanced_sentence_match_score)
    print("Multinomial Naive Bayes Score:", multinomial_naive_bayes_score_value)
    print("Semantic Similarity Score:", semantic_similarity_value)
    print("Coherence Score:", coherence_value)
    print("Relevance Score:", relevance_value)

    return rounded_score
```

## Contributing 🤝

We welcome contributions from the community! Please fork the repository and submit pull requests for any enhancements or bug fixes.

## License 📝

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Thank you for checking out the Answer Evaluation System! If you have any questions or feedback, please feel free to reach out. Happy coding! 🚀
