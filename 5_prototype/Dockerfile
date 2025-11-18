FROM python:3.9

# Step 2: Set the working directory in the container
WORKDIR /app

COPY . .

# Step 4: Install the required Python packages
RUN pip install llama-cpp-python
RUN pip install -r requirements.txt

# Step 5: Expose the default port Streamlit runs on
EXPOSE 8501

ENV browser.gatherUsageStats="false"

# Step 7: Run Streamlit when the container launches
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]