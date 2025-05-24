import React from 'react';
import { Link } from 'react-router-dom';

const NotFoundPage = () => {
  return (
    <div className="container" style={{ textAlign: 'center', padding: '60px 0' }}>
      <h1 style={{ fontSize: '4rem', color: '#3498db', marginBottom: '20px' }}>404</h1>
      <h2>Page Not Found</h2>
      <p className="mb-4">
        The page you are looking for doesn't exist or has been moved.
      </p>
      <Link to="/" className="button">
        <i className="fas fa-home mr-2"></i> Return to Home
      </Link>
    </div>
  );
};

export default NotFoundPage;
