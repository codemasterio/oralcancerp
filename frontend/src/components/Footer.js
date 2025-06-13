import React from 'react';
import { Link } from 'react-router-dom';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="footer">
      <div className="container footer-container">
        <div className="footer-copyright">
          &copy; {currentYear} Oral Cancer Detection. All rights reserved.
        </div>
        <ul className="footer-links">
          <li>
            <Link to="/" className="footer-link">Home</Link>
          </li>
          <li>
            <Link to="/predict" className="footer-link">Predict</Link>
          </li>
          <li>
            <Link to="/about" className="footer-link">About</Link>
          </li>
          <li>
            <Link to="/privacy-policy" className="footer-link">Privacy Policy</Link>
          </li>
          <li>
            <Link to="/terms" className="footer-link">Terms of Service</Link>
          </li>
        </ul>
      </div>
    </footer>
  );
};

export default Footer;
