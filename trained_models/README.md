### **1. `FNN`**
- **Description**: Contains all models trained of the FNN and their performance metrics. Directories are named after the features used to train the certain model.

---

### **2. `LSTM`**
- **Description**: Contains all models trained of the LSTM and their performance metrics. Directories are named after the features used to train the certain model.

---

### **3. `best_model.keras`**
- **Description**: Best performing model. Taken from the LSTM model which has no features ablated. Listed out here individually for ease of use and access.

---

### **3. `encoders.pkl`**
- **Description**: Encoders for feature encoding. The encoding logic should be the same across all models, which means that these encoders can be used for any model (including all LSTM, FNN, HMM models). Listed out here individually for ease of use and access.

---