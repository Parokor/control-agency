#!/usr/bin/env python3
"""
Script para ejecutar las pruebas unitarias.
"""

import os
import sys
import unittest
import argparse

def run_tests(test_module=None, verbose=False):
    """
    Ejecuta las pruebas unitarias.
    
    Args:
        test_module: Módulo de pruebas a ejecutar.
        verbose: Si es True, muestra información detallada.
    
    Returns:
        bool: True si todas las pruebas pasan, False en caso contrario.
    """
    # Configurar el descubrimiento de pruebas
    if test_module:
        # Ejecutar un módulo de pruebas específico
        test_suite = unittest.defaultTestLoader.loadTestsFromName(test_module)
    else:
        # Ejecutar todas las pruebas
        test_suite = unittest.defaultTestLoader.discover('tests')
    
    # Configurar el runner
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    # Ejecutar las pruebas
    result = runner.run(test_suite)
    
    # Devolver True si todas las pruebas pasan
    return result.wasSuccessful()

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Ejecutar pruebas unitarias")
    parser.add_argument("--module", help="Módulo de pruebas a ejecutar")
    parser.add_argument("--verbose", action="store_true", help="Mostrar información detallada")
    
    args = parser.parse_args()
    
    # Ejecutar las pruebas
    success = run_tests(args.module, args.verbose)
    
    # Salir con el código de estado adecuado
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
