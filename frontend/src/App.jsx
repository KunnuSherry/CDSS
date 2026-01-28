import { Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout.jsx'
import './App.css'
import { Landing } from './pages/Landing.jsx'
import { Admin } from './pages/Admin.jsx'
import { Doctor } from './pages/Doctor.jsx'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/admin" element={<Admin />} />
        <Route path="/doctor" element={<Doctor />} />
      </Routes>
    </Layout>
  )
}
