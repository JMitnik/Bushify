import React, { Component } from 'react';
import propTypes from 'prop-types';

class PhotoPolaroid extends Component {
    static propTypes = {
        photo_url: propTypes.string
    };

    renderImage = () => {
        if (this.props.photo_url) {
            return <img src={this.props.photo_url} alt="Spotted"/>
        }
    }

    render() {
        return (
            <div className="box">
                {this.renderImage()}
            </div>
        );
    }
}

export default PhotoPolaroid;
