## File Descriptions

### **1. `separate_object_referenced.py`**
- **Purpose**: For utterances that mention more than 1 object, record them as separate entries.
- **Inputs**: [Coding]
- **Outputs**: [Coding Isolated]

---

### **2. `bound_cog_status.py`**
- **Purpose**: Temporarily bound cognitive statuses by changing the newly mentioned object to the where the object acutally bounds.
- **Inputs**: [Cog Status]
- **Outputs**: [Cog Status Bounded]

---

### **3. `ensure_matching_status.py`**
- **Purpose**: Ensures that all objects which have entered/exited a certain cognitive status have also exited/entered that status. This is the reason why temporary bounding of cognitive statuses exists (since it simplifies the process to keep track of the objects).
- **Inputs**: [Cog Status Bounded]
- **Outputs**: [Terminal text output indicating the entries that do not satisfy the expected condition]

---

### **4. `compare_data.py`**
- **Purpose**: Compares and outputs the differences between initially raw data and data after manual cleaning.
- **Inputs**: [Coding Raw && Coding, Cog Status Raw && Cog Status]
- **Outputs**: [Differences]

---

### **5. `reformat_cog_status.py`**
- **Purpose**: Change the recorded cognitive status of an entry containing multiple cognitive statuses to the highest one. In addition, instead of tracking an object's cognitive status through the columns 'in' and 'out', only display the updated cognitive status at the entry when there is a change.
- **Inputs**: [Cog Status]
- **Outputs**: [Cog Status (Cur Status)]

---

### **6. `create_master_dataset.py`**
- **Purpose**: Creates the master dataset for each individual experiment trials.
- **Inputs**: [Coding Isolated && Cog Status (Cur Status) && Objects]
- **Outputs**: [Master Dataset]

---

### **7. `grammar_role_preprocessing.py`**
- **Description**: A helper script to extract the grammatical role of each object from the utterances.

---

### **8. `true_testing_data.txt`**
- **Description**: A selected set of trials/entries for testing a trained model.

---