document.addEventListener('DOMContentLoaded', function () {
    const svgContainer = document.getElementById('svg-container');

    // Specify the directory path where your SVG files are located
    const directoryPath = '/data/projects/project_alpha/data/processed_data/plots';

    // Fetch the list of SVG files
    fetchSVGFiles(directoryPath);
    
    function fetchSVGFiles(path) {
        fetch(`${path}`)
            .then(response => response.json())
            .then(data => displaySVGFiles(data))
            .catch(error => console.error('Error fetching SVG files:', error));
    }

    function displaySVGFiles(files) {
        files.forEach(file => {
            if (file.endsWith('.svg')) {
                const svgItem = document.createElement('div');
                svgItem.classList.add('svg-item');

                const objectTag = document.createElement('object');
                objectTag.type = 'image/svg+xml';
                objectTag.data = `${directoryPath}${file}`;

                svgItem.appendChild(objectTag);
                svgContainer.appendChild(svgItem);
            }
        });
    }
});
