import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/PrivacyPolicy.css';

const PrivacyPolicy = () => {
  return (
    <div className="privacy-policy">
      <div className="container">
        <Link to="/" className="back-link">← Back to Home</Link>
        <h1>Privacy Policy</h1>
        <div className="content">
          <p>Last updated: June 13, 2025</p>
          
          <section>
            <h2>1. Information We Collect</h2>
            <p>
              We collect the following types of information when you use our service:
            </p>
            <ul>
              <li>Images you upload for analysis</li>
              <li>Analysis results and predictions</li>
              <li>Basic usage data (e.g., timestamps, feature usage)</li>
            </ul>
          </section>

          <section>
            <h2>2. How We Use Your Information</h2>
            <p>We use the collected information to:</p>
            <ul>
              <li>Provide and improve our services</li>
              <li>Process your image analysis requests</li>
              <li>Monitor and analyze service usage</li>
              <li>Prevent, detect, and address technical issues</li>
            </ul>
          </section>

          <section>
            <h2>3. Data Security</h2>
            <p>
              We implement appropriate security measures to protect your data. However,
              please be aware that no method of transmission over the internet is 100% secure.
            </p>
          </section>

          <section>
            <h2>4. Changes to This Policy</h2>
            <p>
              We may update our Privacy Policy from time to time. We will notify you of any changes
              by posting the new Privacy Policy on this page.
            </p>
          </section>

          <section>
            <h2>5. Contact Us</h2>
            <p>
              If you have any questions about this Privacy Policy, please contact us at:
              <br />
              <a href="mailto:privacy@oralcancerdetection.com">privacy@oralcancerdetection.com</a>
            </p>
          </section>
        </div>
      </div>
    </div>
  );
};

export default PrivacyPolicy;
