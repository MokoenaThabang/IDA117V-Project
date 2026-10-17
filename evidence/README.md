# Form evidence

These files were captured on 17 October 2026 from the Streamlit process already serving `app/app.py` at `http://127.0.0.1:8501`. They are screenshots of that running page. They are not drawings and they were not assembled in an image editor.

| File | What it shows |
| --- | --- |
| `screenshots/01_blank_form.png` | The form before any field is chosen. |
| `screenshots/02_completed_sample.png` | The same form after **Fill demonstration sample**. The page showed No Power, PHILIPPI, ward 000, Sub-Council 16, and 12 June 2024. |
| `screenshots/03_prediction.png` | The estimate after that sample was submitted. The page showed Resolved Fast and predicted probabilities 0.598, 0.247, and 0.154. |
| `screenshots/04_validation_error.png` | A blank form after **Estimate resolution outcome**. The page showed the message "Choose a complaint type." |

## Checklist for a later capture

Use the running application. Do not replace these files with a mock-up.

1. Start `streamlit run app/app.py` and open the local URL.
2. Save the untouched form.
3. Press **Fill demonstration sample** and save the filled form before estimating.
4. Press **Estimate resolution outcome** and save the outcome and the probability table.
5. Reload the page, press **Estimate resolution outcome** with the fields empty, and save the validation message.
6. Keep the files only if the text on the page matches what the file name claims.
