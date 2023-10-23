import React, { useEffect } from 'react';
import { get } from './utils/requests';

function App() {

  useEffect(() => {

    /**
     * Example call to Flask
     * @see /src/utils/requests.js
     * @see /app.py
     */
    setTimeout(() => get(
      'example', // Route
      (response) => alert(response), // Response callback
      (error) => console.error(error) // Error callback
    ), 1000);
  }, []);

  return (
    <div>This is a test</div>
  );
}

export default App;
