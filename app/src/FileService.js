const URL = "http://127.0.0.1:5000/upload_file";

const FileService = {
    uploadFile(file, url) {
        var formData = new FormData();
        formData.append('image', file);

        return fetch(URL, {
            method: 'PUT',
            body: formData
        }).then(
            response => response.json()
        )
    }
}

export default FileService;
