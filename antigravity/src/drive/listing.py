from src.drive.auth import get_service

def list_files(folder_id, mime_type='image/tiff'):
    """Lists files in a specific Google Drive folder."""
    service = get_service()
    results = []
    page_token = None

    query = f"'{folder_id}' in parents and mimeType contains '{mime_type}' and trashed = false"

    while True:
        response = service.files().list(q=query,
                                          spaces='drive',
                                          fields='nextPageToken, files(id, name, size, createdTime, mimeType)',
                                          pageToken=page_token).execute()
        for file in response.get('files', []):
            results.append(file)
        
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
            
    return results

def get_file_metadata(file_id):
    service = get_service()
    return service.files().get(fileId=file_id, fields='id, name, mimeType, size, createdTime').execute()
