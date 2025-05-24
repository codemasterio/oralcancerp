import React from 'react';
import { NavLink } from 'react-router-dom';

const Header = () => {
  return (
    <header className="header">
      <div className="container header-container">
        <div className="logo">
          <h1>Oral Cancer Detection</h1>
        </div>
        <nav>
          <ul className="nav-menu">
            <li className="nav-item">
              <NavLink 
                to="/" 
                className={({ isActive }) => 
                  isActive ? "nav-link active" : "nav-link"
                }
              >
                Home
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink 
                to="/predict" 
                className={({ isActive }) => 
                  isActive ? "nav-link active" : "nav-link"
                }
              >
                Predict
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink 
                to="/about" 
                className={({ isActive }) => 
                  isActive ? "nav-link active" : "nav-link"
                }
              >
                About
              </NavLink>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;
