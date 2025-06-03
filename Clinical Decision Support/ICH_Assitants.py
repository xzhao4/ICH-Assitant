import openai
from openai import OpenAI
from openai.types.beta.threads.message_create_params import Attachment, AttachmentToolFileSearch
import json
import os
import time


client = OpenAI(api_key="sk-")  


default_model = 'gpt-4o-2024-11-20'


def create_assistant():
    assistant = client.beta.assistants.create(
        model=default_model,
        instructions="""
You are a medical assistant specializing in the diagnosis and treatment of intracranial hemorrhage (ICH). Your task is to provide structured, precise clinical recommendations **based solely on CT imaging data**, which will be provided in a structured JSON format. You do not have access to any additional clinical data.

You are knowledgeable about the following principles for identifying causes:
- Deep intraparenchymal hemorrhage → Spontaneous ICH
- Subdural or epidural hemorrhage → Traumatic brain injury
- Isolated subarachnoid hemorrhage (SAH), without other types of hemorrhage → Aneurysmal hemorrhage

When formulating recommendations, adhere to the following clinical guidelines and trial results:
- **Spontaneous ICH**: 2022 AHA/ASA Guideline for Spontaneous Intracerebral Hemorrhage, and clinical trials including ENRICH, INTERACT3, INTERACT4, SWITCH, ANNEXA-I.
- **Traumatic brain injury**: 2018 Brain Trauma Foundation Guideline – "Management of Severe Traumatic Brain Injury (First 24 Hours)"
- **Aneurysmal SAH**: 2023 AHA/ASA Guideline for Aneurysmal Subarachnoid Hemorrhage

---

### Step 1: Determine Hemorrhage Cause

Based on the CT findings (location and type of hemorrhage), identify the most likely cause from:
- Spontaneous ICH
- Traumatic brain injury
- Aneurysmal hemorrhage

---

### Step 2: Generate Recommendations

Based on the CT findings, hemorrhage cause, and relevant guidelines, provide:

1. **Additional Examinations**
   - Identify further tests that help complete the diagnostic process, especially focusing on aneurysm detection if applicable.
   - Clearly explain the **purpose** of each test.
   - Specify the **guideline or clinical trial** source supporting each test.

2. **Treatments**
   - Provide specific treatment suggestions (e.g., medical therapy, surgical intervention, BP control).
   - Each treatment should include a **detailed description** of what to do and **why** it is appropriate based on CT findings and the hemorrhage type.
   - Specify the **guideline or trial source** for each recommendation.

---

### Output Format

Respond with both:

#### 1. A JSON object:

```json
{
  "Cause": ,
  "Examinations": [
    {
      "Name": "Test A",
      "Purpose": "Why test A",
      "Source": 
    }
  ],
  "Treatments": [
    {
      "Name": "Treatment A",
      "Purpose": "Detailed treatment plan: what to do and why, based on CT findings",
      "Source": 
    }
  ]
}

2. A **human-readable markdown table** including:

### Examination Recommendations

| Name | Purpose | Source |

### Treatment Recommendations

| Name | Purpose (Detailed Treatment Plan) | Source |
""",
        tools=[{"type": "file_search"}],
        name='ICH Assistant',
    )
    return assistant

def upload_guidelines():
    vector_store = client.beta.vector_stores.create(name="guidelines")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(base_dir, "pdf")
    file_paths = [
        os.path.join(pdf_dir, "Clinical_Trials.pdf"),
        os.path.join(pdf_dir, "2018_TBI_Guideline.pdf"),
        os.path.join(pdf_dir, "2022_ICH_Guideline.pdf"),
        os.path.join(pdf_dir, "2023_aSAH_Guideline.pdf")
    ]
    pdf_files = []
    for path in file_paths:
        if not os.path.exists(path):
            raise Exception(f"Missing required PDF: {path}")
        with open(path, "rb") as f:
            pdf_files.append((os.path.basename(path), f.read()))
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=[(name, content) for name, content in pdf_files]
    )
    if file_batch.status != "completed":
        raise Exception(f"File upload incomplete, status: {file_batch.status}")
    time.sleep(5)
    vs_check = client.beta.vector_stores.retrieve(vector_store_id=vector_store.id)
    return vector_store

def write_error_log(error_log_file, context, error_msg):
    with open(error_log_file, "a", encoding="utf-8") as log_file:
        log_file.write(f"Context: {context}, Error message: {error_msg}\n")

def process_patients(assistant_id, input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    error_log_file = os.path.join(output_dir, "error_log.txt")
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Cannot read input JSON: {e}")
        return
    for patient in data:
        patient_id = patient.get("ID", "unknown")
        prompt = f"""Below is the CT image analysis for a patient:
1. Intraparenchymal Hemorrhage:
   - Volume: {patient['Intraparenchymal_Hemorrhage']['Volume']} ml
   - Location: {patient['Intraparenchymal_Hemorrhage']['Location']}
2. Intraventricular Hemorrhage:
   - Volume: {patient['Intraventricular_Hemorrhage']['Volume']} ml
   - Location: {patient['Intraventricular_Hemorrhage']['Location']}
3. Perihematomal Edema:
   - Volume: {patient['Perihematomal_Edema']['Volume']} ml
   - Location: {patient['Perihematomal_Edema']['Location']}
4. Subarachnoid Hemorrhage:
   - Volume: {patient['Subarachnoid_Hemorrhage']['Volume']} ml
   - Location: {patient['Subarachnoid_Hemorrhage']['Location']}
5. Subdural Hemorrhage:
   - Volume: {patient['Subdural_Hemorrhage']['Volume']} ml
   - Location: {patient['Subdural_Hemorrhage']['Location']}
6. Epidural Hemorrhage:
   - Volume: {patient['Epidural_Hemorrhage']['Volume']} ml
   - Location: {patient['Epidural_Hemorrhage']['Location']}
7. Hydrocephalus: {patient['Hydrocephalus']}
8. Midline Shift: {patient['Midline_Shift']}
"""
        try:
            thread = client.beta.threads.create()
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role='user',
                content=prompt
            )
            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant_id,
                timeout=600,
                temperature=0.1
            )
            if run.status != "completed":
                raise Exception(f'Run failed: {run.status}')
            messages_cursor = client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
            messages = [message for message in messages_cursor]
            treatment_plan = messages[0].content[0].text.value
            output_file = os.path.join(output_dir, f"patient_{patient_id}_report.txt")
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(treatment_plan)
            print(f"Saved report for patient {patient_id}")
        except Exception as ex:
            error_msg = f"Error: {ex}"
            write_error_log(error_log_file, patient_id, error_msg)
            print(f"Error processing patient {patient_id}: {ex}")

def main():
    try:
        assistant = create_assistant()
        vector_store = upload_guidelines()
        if vector_store and vector_store.id:
            client.beta.assistants.update(
                assistant_id=assistant.id,
                tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
            )
        base_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(base_dir, "json", "patients_457.json")
        output_dir = os.path.join(base_dir, "results")
        process_patients(assistant.id, input_file, output_dir)
        print("Processing completed.")
    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    main()