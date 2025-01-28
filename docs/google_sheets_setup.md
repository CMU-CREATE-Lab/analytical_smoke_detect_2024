# Setting Up Google Cloud Project for Sheets

1. Go to the Google Cloud Console and create a new project.
2. Under APIs & Services, enable “Google Sheets API”.
3. In "Credentials", create a service account without any roles.
4. Download the JSON key file and save it in your project directory.
5. Add the file path to your Python “ServiceAccountCredentials.from_json_keyfile_name” function.

