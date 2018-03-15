import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import ImageCanvas from './ImageCanvas';

class App extends Component {

  state = {
    setFile: [],
    displayedImage: null
  }

  handleFileUpload = (event) => {
    this.clearFiles();

    var formData = new FormData();
    var fileField = event.target.files;
    formData.append('test', fileField[0]);

    fetch('http://127.0.0.1:5000/test', {
            method: 'PUT',
            body: formData
        })
        .then(response => response.json())
        .catch(error => console.error('Error:', error))
        .then(response => this.setState({
          displayedImage: response.image_url
        }));
  }

  clearFiles = () => {
    this.setState({
        files: []
    })
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h1 className="App-title">Bushify</h1>
        </header>
        <div>
          <input onChange={e => this.handleFileUpload(e)} type="file" id="input"/>
        </div>
        <ImageCanvas imageUrl={this.state.displayedImage}></ImageCanvas>
      </div>
    );
  }
}

export default App;
