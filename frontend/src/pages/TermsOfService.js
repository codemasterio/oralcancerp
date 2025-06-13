import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/TermsOfService.css';

const TermsOfService = () => {
  return (
    <div className="terms-of-service">
      <div className="container">
        <Link to="/" className="back-link">← Back to Home</Link>
        <h1>Terms of Service</h1>
        <div className="content">
          <p>Last updated: June 13, 2025</p>
          
          <section>
            <h2>1. Acceptance of Terms</h2>
            <p>
              By accessing or using our service, you agree to be bound by these Terms of Service.
              If you disagree with any part of the terms, you may not access the service.
            </p>
          </section>

          <section>
            <h2>2. Use of Service</h2>
            <p>You agree to use the service only for lawful purposes and in accordance with these Terms.</p>
            <p>You must not use the service:</p>
            <ul>
              <li>In any way that violates any applicable law or regulation</li>
              <li>To exploit, harm, or attempt to exploit or harm minors</li>
              <li>To transmit any malicious code or viruses</li>
              <li>To impersonate or attempt to impersonate the company or other users</li>
            </ul>
          </section>

          <section>
            <h2>3. Medical Disclaimer</h2>
            <p>
              The service is not intended to be a substitute for professional medical advice, diagnosis, or treatment.
              Always seek the advice of your physician or other qualified health provider with any questions you may have
              regarding a medical condition.
            </p>
          </section>

          <section>
            <h2>4. Limitation of Liability</h2>
            <p>
              In no event shall we be liable for any indirect, incidental, special, consequential, or punitive damages,
              including without limitation, loss of profits, data, use, goodwill, or other intangible losses, resulting from:
            </p>
            <ul>
              <li>Your access to or use of or inability to access or use the service</li>
              <li>Any conduct or content of any third party on the service</li>
              <li>Any content obtained from the service</li>
            </ul>
          </section>

          <section>
            <h2>5. Changes to Terms</h2>
            <p>
              We reserve the right to modify or replace these terms at any time. We will provide notice of any changes
              by posting the updated terms on this page.
            </p>
          </section>

          <section>
            <h2>6. Contact Us</h2>
            <p>
              If you have any questions about these Terms of Service, please contact us at:
              <br />
              <a href="mailto:legal@oralcancerdetection.com">legal@oralcancerdetection.com</a>
            </p>
          </section>
        </div>
      </div>
    </div>
  );
};

export default TermsOfService;
