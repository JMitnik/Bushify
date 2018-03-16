import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import propTypes from 'prop-types';
import Spinner from './Spinner';
import PhotoPolaroid from './PhotoPolaroid';
import FileService from './FileService';

class ImageCanvas extends Component {
    state = {
        isLoading: false,
        stagedFile: null
    };

    handleFileUpload = (event) => {
        event.preventDefault();
        const file = event.target.files[0];

        this.setState({isLoading: true});

        FileService.uploadFile(file)
            .then(res => {
                this.setState({
                    isLoading: false,
                    stagedFile: res.image_url
                });
            });
    }

    renderSpinner = () => {
        if (this.state.isLoading) {
            return <Spinner></Spinner>;
        }
    }

    renderPrompt = () => {
        if (!this.state.stagedFile) {
            return (
                <div className="container-sm">
                    <div className="box">
                        <div className="box-heading">
                            <h3>Please upload your image</h3>
                        </div>
                        <div className="box-body">
                            Drag your image to this canvas to upload your image.

                            <input className="mg-t-md" type="file" onChange={e => this.handleFileUpload(e)}/>
                        </div>
                    </div>
                </div>
            );
        }
    }

    renderPolaroid = () => {
        if (this.state.stagedFile) {
            return <PhotoPolaroid photo_url={this.state.stagedFile}></PhotoPolaroid>;
        }
    }

    render() {
        return (
            <div className="center">
                {this.renderSpinner()}
                {this.renderPrompt()}
                {this.renderPolaroid()}
            </div>
        );
    }
}

export default ImageCanvas;