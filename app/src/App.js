import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import ImageCanvas from './ImageCanvas';
import './styles/main.scss';

class App extends Component {

  render() {
    return (
      <div className="App">
        <header className="header">
          <h1>Bushify</h1>
        </header>
        <ImageCanvas></ImageCanvas>
      </div>
    );
  }
}

export default App;
